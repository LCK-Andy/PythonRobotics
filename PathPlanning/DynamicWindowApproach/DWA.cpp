#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <array>
#include <tuple>
#include <string>
#include "matplotlibcpp.h" // External library for plotting (install matplotlib-cpp)

namespace plt = matplotlibcpp;

typedef std::array<double, 2> Position;                                     // Represents a 2D position [x, y]
typedef std::vector<std::vector<double>> Trajectory;                        // Represents a trajectory, where each entry is [x, y, yaw, v]
typedef std::vector<double> DynamicWindow;                                  // Represents the dynamic window [v_min, v_max, yaw_rate_min, yaw_rate_max]
typedef std::vector<Position> Obstacles;                                    // Represents a list of obstacles as [x, y]
typedef std::tuple<std::array<double, 2>, Trajectory> ControlAndTrajectory; // Represents control input and trajectory


enum class RobotType
{
    Circle,
    Rectangle
};

struct Config
{
    // Robot parameters
    double max_speed = 10.0;                    // [m/s]
    double min_speed = -0.5;                   // [m/s]
    double max_yaw_rate = M_PI;            // [rad/s]
    double max_accel = 0.2;                    // [m/ss]
    double max_delta_yaw_rate = M_PI;    // [rad/ss]
    
    double v_resolution = 0.01;                // [m/s]
    double yaw_rate_resolution = M_PI / 180.0; // [rad/s]
    double dt = 0.1;                           // [s] Time tick for motion prediction
    double predict_time = 2.0;                 // [s]
    double to_goal_cost_gain = 0.2;
    double speed_cost_gain = 1.0;
    double obstacle_cost_gain = 1.0;
    double robot_stuck_flag_cons = 0.001; // Constant to prevent robot from getting stuck
    RobotType robot_type = RobotType::Circle;

    // Robot dimensions
    double robot_radius = 1.0; // [m] for collision check (Circle)
    double robot_width = 0.5;  // [m] for collision check (Rectangle)
    double robot_length = 1.2; // [m] for collision check (Rectangle)

    // Obstacles: [x, y]
    Obstacles obstacles = {
        {-1, -1}, {0, 2}, {4.0, 2.0}, {5.0, 4.0}, {5.0, 5.0}, {5.0, 6.0}, {5.0, 9.0}, {8.0, 9.0}, {7.0, 9.0}, {8.0, 10.0}, {9.0, 11.0}, {12.0, 13.0}, {12.0, 12.0}, {15.0, 15.0}, {13.0, 13.0}};
};

struct State
{
    double x = 0.0;          // [m]
    double y = 0.0;          // [m]
    double yaw = M_PI / 8.0; // [rad]
    double v = 0.0;          // [m/s]
    double omega = 0.0;      // [rad/s]
};

// Function prototypes
/**
 * Simulates the robot's motion based on its current state, control input, and time step.
 * 
 * @param x Current state of the robot.
 * @param u Control input [velocity, yaw_rate].
 * @param dt Time step for the simulation.
 * @return Updated state of the robot after applying the control input for the time step.
 */
State motion(State x, const Position &u, double dt);

/**
 * Calculates the dynamic window, a set of possible velocities and angular velocities, 
 * based on the robot's current state and constraints.
 * 
 * @param x Current state of the robot.
 * @param config Configuration parameters including robot constraints.
 * @return A vector containing the minimum and maximum velocities and yaw rates:
 *         [v_min, v_max, yaw_rate_min, yaw_rate_max].
 */
DynamicWindow calc_dynamic_window(const State &x, const Config &config);

/**
 * Predicts the trajectory of the robot given an initial state, a control input, and time..
 *
 * @param x_init Initial state of the robot.
 * @param v Linear velocity.
 * @param y Yaw rate (angular velocity).
 * @param config Configuration parameters including time step and prediction time.
 * @return A vector of vectors representing the predicted trajectory, where each inner vector contains:
 *         [x_position, y_position, yaw, velocity] at each time step.
 */
Trajectory predict_trajectory(const State &x_init, double v, double omega, const Config &config);

/**
 * Finds the optimal control input (u) and trajectory by evaluating all possible trajectories within the dynamic window.
 * 
 * @param x Current state of the robot.
 * @param dw Dynamic window [v_min, v_max, yaw_rate_min, yaw_rate_max].
 * @param config Configuration parameters including cost gains and robot constraints.
 * @param goal Goal position [x, y].
 * @param ob List of obstacles, each represented as [x, y].
 * @return A tuple containing:
 *         - Best control input [velocity, yaw_rate].
 *         - Best predicted trajectory as a vector of vectors, where each inner vector contains:
 *           [x_position, y_position, yaw, velocity] at each time step.
 */
ControlAndTrajectory calc_control_and_trajectory(
    const State &x, const DynamicWindow &dw, const Config &config, const Position &goal, const Obstacles &ob);

/**
 * Calculates the cost of a trajectory based on its alignment with the goal.
 * 
 * @param trajectory Predicted trajectory as a vector of vectors, where each inner vector contains:
 *                   [x_position, y_position, yaw, velocity].
 * @param goal Goal position [x, y].
 * @return Cost value representing how well the trajectory aligns with the goal.
 */
double calc_to_goal_cost(const Trajectory &trajectory, const Position &goal);

/**
 * Calculates the cost of a trajectory based on its proximity to obstacles.
 * 
 * @param trajectory Predicted trajectory as a vector of vectors, where each inner vector contains:
 *                   [x_position, y_position, yaw, velocity].
 * @param ob List of obstacles, each represented as [x, y].
 * @param config Configuration parameters including robot radius for collision checking.
 * @return Cost value representing the risk of collision with obstacles along the trajectory.
 */
double calc_obstacle_cost(const Trajectory &trajectory, const Obstacles &ob, const Config &config);
void plot_robot(double x, double y, double yaw, const Config &config);
void plot_arrow(double x, double y, double yaw, double length = 0.5, double width = 0.1);

State motion(State x, const Position &u, double dt)
{
    // Updates the robot's orientation (yaw) using the angular velocity.
    x.yaw += u[1] * dt;

    // Updates the robot's position (x, y) using the linear velocity and orientation.
    x.x += u[0] * std::cos(x.yaw) * dt;
    x.y += u[0] * std::sin(x.yaw) * dt;

    // Updates the robot's linear and angular velocities.
    x.v = u[0];
    x.omega = u[1];

    return x;
}

DynamicWindow calc_dynamic_window(const State &x, const Config &config)
{
    // Robot specification limits Vs = [v_min, v_max, yaw_rate_min, yaw_rate_max]
    std::vector<double> Vs = {config.min_speed, config.max_speed,
                              -config.max_yaw_rate, config.max_yaw_rate};
    // Motion limits Vd 
    std::vector<double> Vd = {x.v - config.max_accel * config.dt,
                              x.v + config.max_accel * config.dt,
                              x.omega - config.max_delta_yaw_rate * config.dt,
                              x.omega + config.max_delta_yaw_rate * config.dt};

    // Final dynamic window: [v_min, v_max, yaw_rate_min, yaw_rate_max]
    return {std::max(Vs[0], Vd[0]), std::min(Vs[1], Vd[1]),
            std::max(Vs[2], Vd[2]), std::min(Vs[3], Vd[3])};
}

Trajectory predict_trajectory(const State &x_init, double v, double omega, const Config &config)
{

    State x = x_init;
    Trajectory trajectory = {{x.x, x.y, x.yaw, x.v}};
    double time = 0.0;

    // Simulates the motion of the robot over the prediction time using the motion function.
    while (time <= config.predict_time)
    {
        x = motion(x, {v, omega}, config.dt);

        // Records the state (x, y, yaw, v) at each step.
        trajectory.push_back({x.x, x.y, x.yaw, x.v});
        time += config.dt;
    }

    return trajectory;
}

ControlAndTrajectory calc_control_and_trajectory(
    const State &x, const DynamicWindow &dw, const Config &config, const Position &goal, const Obstacles &ob)
{

    State x_init = x;
    double min_cost = std::numeric_limits<double>::infinity();
    Position best_u = {0.0, 0.0};
    Trajectory best_trajectory = {{x.x, x.y, x.yaw, x.v}};

    // Iterates over all possible velocities (v) and angular velocities (y) in the dynamic window.
    for (double v = dw[0]; v <= dw[1]; v += config.v_resolution)
    {
        for (double y = dw[2]; y <= dw[3]; y += config.yaw_rate_resolution)
        {
            // Predicts the trajectory for each (v, y) pair using predict_trajectory.
            auto trajectory = predict_trajectory(x_init, v, y, config);

            // How well the trajectory aligns with the goal.
            double to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal);

            // Penalizes lower speeds to encourage faster movement.
            double speed_cost = config.speed_cost_gain * (config.max_speed - trajectory.back()[3]);

            // Penalizes trajectories that come close to obstacles.
            double ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config);

            double final_cost = to_goal_cost + speed_cost + ob_cost;

            // Finds the (v, y) pair with the lowest total cost and returns:
            if (final_cost < min_cost)
            {
                min_cost = final_cost;
                best_u = {v, y};
                best_trajectory = trajectory;
            }
        }
    }
    return {best_u, best_trajectory};
}

double calc_to_goal_cost(const Trajectory &trajectory, const Position &goal)
{
    // Computes the angle between the robot's final position and the goal.
    double dx = goal[0] - trajectory.back()[0];
    double dy = goal[1] - trajectory.back()[1];
    double error_angle = std::atan2(dy, dx);

    // Returns the angular difference (cost) between the final orientation of the robot and the direction to the goal.
    double cost_angle = error_angle - trajectory.back()[2];
    return std::abs(std::atan2(std::sin(cost_angle), std::cos(cost_angle)));
}

double calc_obstacle_cost(const Trajectory &trajectory, const Obstacles &ob, const Config &config)
{
    double min_r = std::numeric_limits<double>::infinity();

    // For each point in the trajectory, calculates the distance to each obstacle.
    for (const auto &point : trajectory)
    {
        for (const auto &obstacle : ob)
        {
            // calculate the inverse of the distance to an obstacle (closer obstacles = higher cost).
            double dx = point[0] - obstacle[0];
            double dy = point[1] - obstacle[1];
            double r = std::hypot(dx, dy);

            // If any point is within the robot's radius of an obstacle, assigns an infinite cost (collision).
            if (r <= config.robot_radius)
            {
                return std::numeric_limits<double>::infinity();
            }
            min_r = std::min(min_r, r);
        }
    }

    // Returns the inverse of the smallest distance to an obstacle.
    return 1.0 / min_r;
}

void plot_robot(double x, double y, double yaw, const Config &config)
{
    if (config.robot_type == RobotType::Rectangle)
    {
        // Rectangle robot outline
    }
    else if (config.robot_type == RobotType::Circle)
    {
        plt::plot({x}, {y}, "ob");
    }
}

void plot_arrow(double x, double y, double yaw, double length, double width)
{
    double dx = length * std::cos(yaw);
    double dy = length * std::sin(yaw);
    plt::arrow(x, y, dx, dy, "->");
}

int main()
{
    Config config;
    State x;
    Position goal = {10.0, 10.0};
    Trajectory trajectory = {{x.x, x.y, x.yaw, x.v}};

    while (true)
    {
        DynamicWindow dw = calc_dynamic_window(x, config);
        auto [u, predicted_trajectory] = calc_control_and_trajectory(x, dw, config, goal, config.obstacles);
        x = motion(x, u, config.dt);
        trajectory.push_back({x.x, x.y, x.yaw, x.v});

        plt::clf();
        plt::plot({goal[0]}, {goal[1]}, "xb");
        for (const auto &ob : config.obstacles)
        {
            plt::plot({ob[0]}, {ob[1]}, "ok");
        }
        plot_robot(x.x, x.y, x.yaw, config);
        plt::pause(0.001);

        double dist_to_goal = std::hypot(x.x - goal[0], x.y - goal[1]);
        if (dist_to_goal <= config.robot_radius)
        {
            std::cout << "Goal!!" << std::endl;
            break;
        }
    }

    std::cout << "Done" << std::endl;
    plt::show();
    return 0;
}

/**
 * compile command:
 * g++ -std=c++17 \
-I/usr/include/python3.12 \
-I/usr/lib/python3/dist-packages/numpy/core/include \
-L/usr/lib/python3.12/config-3.12-x86_64-linux-gnu \
DWA.cpp -lpython3.12 -o dwa_robot
 */