struct VehicleCommand
    steering_angle::Float64
    velocity::Float64
    controlled::Bool
end

function get_c()
    ret = ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid},Int32), stdin.handle, true)
    ret == 0 || error("unable to switch to raw mode")
    c = read(stdin, Char)
    ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid},Int32), stdin.handle, false)
    c
end

function keyboard_client(host::IPAddr=IPv4(0), port=4444; v_step = 1.0, s_step = π/10)
    socket = Sockets.connect(host, port)
    (peer_host, peer_port) = getpeername(socket)
    msg = deserialize(socket) # Visualization info
    @info msg

    @async while isopen(socket)
        sleep(0.001)
        state_msg = deserialize(socket)
        measurements = state_msg.measurements
        num_cam = 0
        num_imu = 0
        num_gps = 0
        num_gt = 0
        for meas in measurements
            if meas isa GroundTruthMeasurement
                num_gt += 1
            elseif meas isa CameraMeasurement
                num_cam += 1
            elseif meas isa IMUMeasurement
                num_imu += 1
            elseif meas isa GPSMeasurement
                num_gps += 1
            end
        end
  #      @info "Measurements received: $num_gt gt; $num_cam cam; $num_imu imu; $num_gps gps"
    end
    
    target_velocity = 0.0
    steering_angle = 0.0
    controlled = true
    @info "Press 'q' at any time to terminate vehicle."
    while controlled && isopen(socket)
        key = get_c()
        if key == 'q'
            # terminate vehicle
            controlled = false
            target_velocity = 0.0
            steering_angle = 0.0
            @info "Terminating Keyboard Client."
        elseif key == 'i'
            # increase target velocity
            target_velocity += v_step
            @info "Target velocity: $target_velocity"
        elseif key == 'k'
            # decrease forward force
            target_velocity -= v_step
            @info "Target velocity: $target_velocity"
        elseif key == 'j'
            # increase steering angle
            steering_angle += s_step
            @info "Target steering angle: $steering_angle"
        elseif key == 'l'
            # decrease steering angle
            steering_angle -= s_step
            @info "Target steering angle: $steering_angle"
        end
        cmd = VehicleCommand(steering_angle, target_velocity, controlled)
        serialize(socket, cmd)
    end
end

function example_client(host::IPAddr=IPv4(0), port=4444)
    socket = Sockets.connect(host, port)
    map_segments = training_map()
    (; chevy_base) = load_mechanism()

    @async while isopen(socket)
        state_msg = deserialize(socket)
    end
   
    shutdown = false
    persist = true
    while isopen(socket)
        position = state_msg.q[5:7]
        @info position
        if norm(position) >= 100
            shutdown = true
            persist = false
        end
        cmd = VehicleCommand(0.0, 2.5, persist, shutdown)
        serialize(socket, cmd) 
    end

end

function convert_to_localization_type(gt_meas::GroundTruthMeasurement)
    MyLocalizationType(gt_meas.time, FullVehicleState(gt_meas.position, gt_meas.orientation, gt_meas.velocity, gt_meas.angular_velocity), gt_meas.size)
end

function fake_localize(gt_channel, fake_localize_state_channel, ego_id, quit_channel)
    @info "starting fake localize"
    while !fetch(quit_channel)
        sleep(0.001)
        fresh_gt_meas = []
        while isready(gt_channel)
            meas = take!(gt_channel)
            push!(fresh_gt_meas, meas)
        end

        latest_meas_time = -Inf
        latest_meas = nothing
        for meas in fresh_gt_meas
            if meas.time > latest_meas_time && meas.vehicle_id == ego_id
                latest_meas = meas
                latest_meas_time = meas.time
            end
        end

        # Convert latest_meas to MyLocalizationType
        my_converted_gt_message = convert_to_localization_type(latest_meas)
        @info "Fake localization: $my_converted_gt_message"

        if isready(fake_localize_state_channel)
            take!(fake_localize_state_channel)
        end
        put!(fake_localize_state_channel, my_converted_gt_message)
    end
end

function compare_localization_results(localize_state_channel, fake_localize_state_channel, quit_channel)
    @info "starting compare localize"
    while !fetch(quit_channel)
        sleep(0.001)
        if isready(localize_state_channel) && isready(fake_localize_state_channel)
            estimated_state = take!(localize_state_channel)
            fake_estimated_state = take!(fake_localize_state_channel)

            position_error = norm(estimated_state.x.position - fake_estimated_state.x.position)
            orientation_error = norm(estimated_state.x.quaternion - fake_estimated_state.x.quaternion)
            velocity_error = norm(estimated_state.x.velocity - fake_estimated_state.x.velocity)
            angular_velocity_error = norm(estimated_state.x.angular_velocity - fake_estimated_state.x.angular_velocity)

            @info "Position error: $position_error, Orientation error: $orientation_error, Velocity error: $velocity_error, Angular velocity error: $angular_velocity_error"
        end
    end
end



function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end



function test_client(host::IPAddr=IPv4(0), port=4444, v_step = 1.0, s_step = π/10)
    socket = Sockets.connect(host, port)
    map_segments = VehicleSim.training_map()
    
    msg = deserialize(socket) # Visualization info
    @info msg

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)

    localization_state_channel = Channel{MyLocalizationType}(1)
    fake_localize_state_channel = Channel{MyLocalizationType}(1)
    quit_channel = Channel{Bool}(1)

    target_map_segment = 0 # (not a valid segment, will be overwritten by message)
    ego_vehicle_id = 0 # (not a valid id, will be overwritten by message. This is used for discerning ground-truth messages)

    errormonitor(@async while true
        # This while loop reads to the end of the socket stream (makes sure you
        # are looking at the latest messages)
        sleep(0.001)
        local measurement_msg
        received = false
        while true
            @async eof(socket)
            if bytesavailable(socket) > 0
                measurement_msg = deserialize(socket)
                received = true
            else
                break
            end
        end
        !received && continue
        target_map_segment = measurement_msg.target_segment
        ego_vehicle_id = measurement_msg.vehicle_id
        for meas in measurement_msg.measurements
            if meas isa GPSMeasurement
                !isfull(gps_channel) && put!(gps_channel, meas)
            elseif meas isa IMUMeasurement
                !isfull(imu_channel) && put!(imu_channel, meas)
            elseif meas isa CameraMeasurement
                !isfull(cam_channel) && put!(cam_channel, meas)
            elseif meas isa GroundTruthMeasurement
                !isfull(gt_channel) && put!(gt_channel, meas)
            end
        end
    end)


    errormonitor(@async localize(gps_channel, imu_channel, localization_state_channel, quit_channel))
    errormonitor(@async fake_localize(gt_channel, fake_localize_state_channel, ego_vehicle_id, quit_channel))
    errormonitor(@async compare_localization_results(localization_state_channel, fake_localize_state_channel, quit_channel))
    # @infiltrate
    target_velocity = 0.0
    steering_angle = 0.0
    controlled = true
    @info "Press 'q' at any time to terminate vehicle."
    while controlled && isopen(socket)
        key = get_c()
        if key == 'q'
            # terminate vehicle
            controlled = false
            target_velocity = 0.0
            steering_angle = 0.0
            @info "Terminating test client and its worker threads"
            # sends msg to workers threads to throw error so they die
            # this lets us use Revise to handle code changes
            put!(shutdown_channel, true)
        elseif key == 'i'
            # increase target velocity
            target_velocity += v_step
            @info "Target velocity: $target_velocity"
        elseif key == 'k'
            # decrease forward force
            target_velocity -= v_step
            @info "Target velocity: $target_velocity"
        elseif key == 'j'
            # increase steering angle
            steering_angle += s_step
            @info "Target steering angle: $steering_angle"
        elseif key == 'l'
            # decrease steering angle
            steering_angle -= s_step
            @info "Target steering angle: $steering_angle"
        end
        cmd = VehicleCommand(steering_angle, target_velocity, controlled)
        serialize(socket, cmd)
    end
end
