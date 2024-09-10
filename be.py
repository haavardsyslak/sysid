import blueye.protocol as bp
from blueye.sdk import Drone


def callback_imu_raw(msg_type, msg):
    pass


def callback_depth(msg_type, msg):
    pass


if __name__ == "__main__":
    drone = Drone()
    cb_imu = drone.telemetry.add_msg_callback(
        [bp.Imu1Tel, bp.Imu2Tel], callback_imu_raw
    )
    cb_depth = drone.telemetry.add_msg_callback([bp.DepthTel], callback_depth)

    drone.telemetry.set_msg_publish_frequency([bp.Imu1Tel, bp.Imu2Tel], 50)
    drone.telemetry.set_msg_publish_frequency([bp.DepthTel], 50)

    input()
    drone.telemetry.remove_msg_callback(cb_imu)
    drone.telemetry.remove_msg_callback(cb_depth)
