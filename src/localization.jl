struct SimpleVehicleState
    p1::Float64
    p2::Float64
    θ::Float64
    v::Float64
    l::Float64
    w::Float64
    h::Float64
end

struct FullVehicleState
    position::SVector{3, Float64}
    quaternion::SVector{4, Float64}
    velocity::SVector{3, Float64}
    angular_velocity::SVector{3, Float64}
end

struct MyLocalizationType
    last_update::Float64
    x::FullVehicleState
    size::SVector{3, Float64}
end

function localize(gps_channel, imu_channel, localization_state_channel, shutdown_channel)
    @info "Starting localization"

    # initial state of vehicle
    x0 = zeros(13)
    first_gps = take!(gps_channel)
    @info "First GPS: $first_gps"

    θ = first_gps.heading

    # rotation matrix from segment to world frame
    R = RotZ(θ)

    # get quaternion from rotation matrix
    qw = sqrt(1 + R[1,1] + R[2,2] + R[3,3]) / 2
    qx = (R[3,2] - R[2,3]) / (4 * qw)
    qy = (R[1,3] - R[3,1]) / (4 * qw)
    qz = (R[2,1] - R[1,2]) / (4 * qw)
    x0[4:7] = [qw, qx, qy, qz]

    x0[1:2] = [first_gps.lat, first_gps.long]
    x0[3] = 1.0 
    x = x0
    last_update = 0.0

    P = zeros(13, 13)
    diag_vals = [
        1.0, 1.0, 1.0, 
        0.1, 0.1, 0.1, 0.1, 
        0.1, 0.1, 0.1, 
        0.1, 0.1, 0.1
    ]
    P = diagm(diag_vals)

    # process noise
    Q = 0.1 * I(13)

    # measurement noise for both GPS and IMU
    R_gps = Diagonal([3.0, 3.0, 1.0])
    R_imu = 0.01 * I(6) # 0.01?

    @info "Starting localization loop"

    # Set up algorithm / initialize variables
    while true
        sleep(0.001)
        isready(shutdown_channel) && break
        fresh_gps_meas = []
        while isready(gps_channel)
            meas = take!(gps_channel)
            push!(fresh_gps_meas, meas)
        end
        fresh_imu_meas = []
        while isready(imu_channel)
            meas = take!(imu_channel)
            push!(fresh_imu_meas, meas)
        end

        # process measurements
        while length(fresh_gps_meas) > 0 && length(fresh_imu_meas) > 0
            sleep(0.001)
            # grab measurements and sort them by time
            all_meas = [fresh_gps_meas..., fresh_imu_meas...]
            sort!(all_meas, by = m -> m.time)

            # Process the earliest measurement
            z = all_meas[1]
            Δ = z.time - last_update
            last_update = z.time

            # Remove the processed measurement from the respective list
            if z isa GPSMeasurement
                fresh_gps_meas = fresh_gps_meas[2:end]
            elseif z isa IMUMeasurement
                fresh_imu_meas = fresh_imu_meas[2:end]
            end

            # @info "Processing measurement of type $(typeof(z))"

            # run filter
            if z isa GPSMeasurement
                R = R_gps
            elseif z isa IMUMeasurement
                R = R_imu
            end
            x, P = filter(x, z, P, Q, R, Δ)

            # publish state
            full = FullVehicleState(x[1:3], x[4:7], x[8:10], x[11:13])
            localization_state = MyLocalizationType(last_update, full, [13.2, 5.7, 5.3])
            if isready(localization_state_channel)
                take!(localization_state_channel)
            end
            put!(localization_state_channel, localization_state)
        end
    end
end


# for computing the imu transform
function custom_roty(θ)
    R = zeros(3, 3)
    R = [cos(θ) 0 sin(θ); 0 1 0; -sin(θ) 0 cos(θ)]
    return R
end

function get_imu_transform1()
    R_imu_to_body = custom_roty(0.02)
    t_imu_to_body = [0, 0, 0.7]

    T = [R_imu_to_body t_imu_to_body]
end


# -------------------------------- EKF functions -------------------------------- #
# process model in measurements.jl

# measurement model
function h(x, z)
    if z isa GPSMeasurement
        T = get_gps_transform()
        gps_loc_body = T*[zeros(3); 1.0]
        xyz_body = x[1:3] # position
        q_body = x[4:7] # quaternion

        Tbody = get_body_transform(q_body, xyz_body)
        xyz_gps = Tbody * [gps_loc_body; 1]
        yaw = extract_yaw_from_quaternion(q_body)
        meas = [xyz_gps[1:2]; yaw]

        return meas
    elseif z isa IMUMeasurement
        # convert to body frame
        T_body_imu = get_imu_transform1()
        T_imu_body = invert_transform(T_body_imu)
        R = T_imu_body[1:3,1:3]
        p = T_imu_body[1:3,end]

        v_body = x[8:10]
        ω_body = x[11:13]

        ω_imu = R * ω_body
        v_imu = R * v_body + p × ω_imu

        # need to return an imu frame, which is the linear and angular velocity of the body frame
        return [v_imu; ω_imu]
    else
        error("Unknown measurement type")
    end
end

# jacobian of h with respect to x
function jac_hx(x, z)
    jacobian(x -> h(x, z), x)[1]
end

# convert z to a vector
function z_to_vec(z)
    if z isa GPSMeasurement
        return [z.lat; z.long; z.heading]
    elseif z isa IMUMeasurement
        return [z.linear_vel; z.angular_vel]
    else
        error("Unknown measurement type")
    end
end

# extended kalman filter
function filter(x, z, P, Q, R, Δ)
    # predict
    x̂ = f(x, Δ)
    F = Jac_x_f(x, Δ)
    P̂ = F * P * F' + Q

    # update
    z_vec = z_to_vec(z)

    y = z_vec - h(x̂, z)
    H = jac_hx(x̂, z)
    S = H * P̂ * H' + R
    K = P̂ * H' * inv(S)

    x̂ = x̂ + K * y
    P = (I - K * H) * P̂
    @info "x: $x"

    return x̂, P
end


# -------------------------------- Segment functions -------------------------------- #
function get_cur_segment(position)
    all_segments = training_map()

    for (id, road_segment) in all_segments
        left_lane_boundary = road_segment.lane_boundaries[1]
        right_lane_boundary = road_segment.lane_boundaries[2]

        lx1, ly1 = left_lane_boundary.pt_a[1], left_lane_boundary.pt_a[2]
        lx2, ly2 = left_lane_boundary.pt_b[1], left_lane_boundary.pt_b[2]

        rx1, ry1 = right_lane_boundary.pt_a[1], right_lane_boundary.pt_a[2]
        rx2, ry2 = right_lane_boundary.pt_b[1], right_lane_boundary.pt_b[2]

        xmin = min([lx1, lx2, rx1, rx2]...)
        xmax = max([lx1, lx2, rx1, rx2]...)
        ymin = min([ly1, ly2, ry1, ry2]...)
        ymax = max([ly1, ly2, ry1, ry2]...)

        x = position[1]
        y = position[2]

        if (x <= xmax && x >= xmin && y <= ymax && y >= ymin)
            @info "id: $id"
            return id
        end
    end

    error("No segment found")
end

