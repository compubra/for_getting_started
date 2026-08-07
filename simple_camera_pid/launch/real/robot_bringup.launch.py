"""Bring up the camera driver, the TurtleBot3 base driver, and the vision
debug overlay together, on this robot (the Raspberry Pi).

Replaces the two-step ``ros2 run rclcpp_components component_container`` +
``ros2 component load ...`` shell dance and the separate ``ros2 run
turtlebot3_node turtlebot3_ros`` invocation (previously in ``~/start_robot.sh``)
with one launch file. Two things this deliberately does NOT do, matching that
script:

  - No lidar (LDS driver): this robot has none mounted, and
    ``turtlebot3_bringup``'s own ``robot.launch.py`` fails outright without
    one (it unconditionally includes an LDS launch file). Skip it entirely
    rather than route around a hardware dependency that isn't there.
  - No ``image_view`` composable node alongside the camera: camera_ros's own
    ``camera.launch.py`` adds one unconditionally when the package is
    installed, which has no display to attach to on this headless Pi and
    crashes the whole component container. This launch file loads only
    ``camera::CameraNode`` into a bare container.

``vision_debug_node`` runs by default (``vision_debug:=true``) -- per its own
module docstring and real_monitor.launch.py's, it has to run ON THE PI (next
to the camera) rather than on the PC, since subscribing to the raw camera
feed remotely over WiFi is the exact bandwidth problem this whole split
exists to avoid. Reads the same ``real_line_follower.yaml`` the actual
controller nodes use (via ``vision_debug_config_file``) so its ROI/vision
tuning can never drift from what the robot is actually running, and publishes
the annotated overlay on ``/vision_debug/image_raw`` (huge, uncompressed --
do not subscribe to this over WiFi) and ``/vision_debug/image_raw/compressed``
(the one to view remotely, e.g. in rviz2's Image display or real_monitor
.launch.py on the PC).

Usage
-----
    ros2 launch simple_camera_pid robot_bringup.launch.py
    ros2 launch simple_camera_pid robot_bringup.launch.py width:=320 height:=240
    ros2 launch simple_camera_pid robot_bringup.launch.py jpeg_quality:=50
    ros2 launch simple_camera_pid robot_bringup.launch.py usb_port:=/dev/ttyACM1
    ros2 launch simple_camera_pid robot_bringup.launch.py vision_debug:=false
    ros2 launch simple_camera_pid robot_bringup.launch.py orientation:=0
        (only if the camera is ever remounted right-side-up -- see
        orientation's own entry below; default is already 180 for this
        robot's current upside-down mount)
    ros2 launch simple_camera_pid robot_bringup.launch.py scaler_crop:="x,y,w,h"
        (syntax only -- UNCALIBRATED placeholder, do not copy real numbers
        from here; see scaler_crop's own entry below before using this)

Arguments
---------
    width                   Camera capture *output* width in pixels (default
                          640) -- what gets published on /camera/image_raw,
                          after libcamera scales the (possibly already
                          scaler_crop-cropped) sensor region down to this
                          size. Independent of scaler_crop's own units (see
                          below).
    height                  Camera capture *output* height in pixels
                          (default 480). Same relationship to scaler_crop as
                          width above.
    jpeg_quality            /camera/image_raw/compressed JPEG quality, 0-100
                          (default 70 -- see camera_ros_ws bandwidth notes:
                          640x480 at quality 95 saturates a WiFi link at
                          ~3.4 MB/s, quality 70 cuts that to ~0.9 MB/s for a
                          similar picture).
    usb_port                OpenCR serial port for the base driver (default
                          /dev/ttyACM0).
    tb3_param_dir           TurtleBot3 kinematics/motor parameter YAML
                          (default turtlebot3_bringup's burger.yaml -- change
                          this if this robot is ever not a Burger).
    vision_debug            Launch vision_debug_node (default true).
    vision_debug_config_file
                            Vision/camera tuning YAML for vision_debug_node
                          (default config/real/real_line_follower.yaml, same
                          file the real controller nodes read).
    orientation              camera_ros/libcamera "orientation" control: 0, 90,
                          180, or 270 degrees (default 180 -- this robot's
                          camera is physically mounted upside-down, confirmed
                          2026-08-07; libcamera rotates the sensor readout
                          itself before any output scaling/cropping, so this
                          is independent of width/height/scaler_crop below).
                          read_only in camera_ros -- takes effect only at
                          node startup. Pass orientation:=0 if the mount is
                          ever corrected/reversed.
    scaler_crop             libcamera ScalerCrop control: "x,y,w,h", in the
                          IMX219 sensor's native full-resolution pixel space
                          (3280x2464 per the mode list camera_ros reports at
                          startup -- NOT this launch file's width/height
                          output-scaling args), selecting which region of the
                          sensor is read out before scaling down to
                          width/height for output. Empty string (default) =
                          unset = full sensor region, current behavior
                          unchanged.
                          UNCALIBRATED as of 2026-08-06: added to let the
                          sensor read out a NARROWER, LOWER region of its
                          full view at higher pixel density (motivation: on
                          a real run, the robot's camera lost sight of the
                          line entirely mid-curve -- confirmed by the
                          operator watching, not a detection bug -- because
                          the line swung further out of the current
                          640x480 uncropped view than the fixed
                          roi_bottom_fraction/camera geometry anticipates).
                          Does NOT work out of the box: (1) the actual x/y/w/h
                          below is a placeholder, not measured against this
                          camera's real mount angle/height: reason through it
                          from the mode list's full sensor size and this
                          robot's camera_pitch_deg/camera_mount_height, or
                          just sweep values against vision_debug_node's live
                          overlay; (2) common/camera_geometry.py's
                          turtlebot3_burger_real_camera() (camera_fovy_deg
                          particularly) assumes the published frame is an
                          UNCROPPED full sensor view scaled to width/height --
                          cropping the sensor read-out narrows the true FOV
                          those numbers describe, so they need to be
                          recalculated for whatever crop region actually gets
                          used, not left at their current (already only
                          approximately measured) values. Do not trust a
                          non-empty value here without re-deriving both.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

# Matches ~/.bashrc: export ROS_DOMAIN_ID=30 / TURTLEBOT3_MODEL=burger. Only
# warns (rather than forcing either) since this launch file's own child
# processes inherit whatever's already in this shell's environment -- a
# forced override here wouldn't fix a mismatched PC-side shell anyway, and
# silently rewriting env vars out from under the user is more surprising
# than useful.
EXPECTED_ROS_DOMAIN_ID = '30'
EXPECTED_TURTLEBOT3_MODEL = 'burger'


def _warn_if_env_mismatch(context, *args, **kwargs):
    messages = []
    domain_id = os.environ.get('ROS_DOMAIN_ID')
    if domain_id != EXPECTED_ROS_DOMAIN_ID:
        messages.append(LogInfo(msg=(
            f"** ROS_DOMAIN_ID={domain_id!r} in this shell, expected "
            f"{EXPECTED_ROS_DOMAIN_ID!r} (see ~/.bashrc). If a PC on the same "
            "network is not on the same ROS_DOMAIN_ID, it will never see "
            "this robot's topics. **"
        )))
    model = os.environ.get('TURTLEBOT3_MODEL')
    if model != EXPECTED_TURTLEBOT3_MODEL:
        messages.append(LogInfo(msg=(
            f"** TURTLEBOT3_MODEL={model!r} in this shell, expected "
            f"{EXPECTED_TURTLEBOT3_MODEL!r} (see ~/.bashrc). turtlebot3_node "
            "itself doesn't read this var, but other TurtleBot3 tooling "
            "does -- set it to avoid surprises elsewhere. **"
        )))
    return messages


def _make_camera_container(context, *args, **kwargs):
    """Built via OpaqueFunction (rather than a static ComposableNodeContainer
    in the top-level LaunchDescription list) only because ScalerCrop needs
    its actual runtime string value to decide whether to include the
    parameter at all -- an empty scaler_crop must result in camera_ros never
    seeing a ScalerCrop key, not an empty/invalid one, to keep default
    (uncropped) behavior byte-for-byte unchanged. See scaler_crop's docstring
    above before setting it to anything.
    """
    width = LaunchConfiguration('width')
    height = LaunchConfiguration('height')
    jpeg_quality = LaunchConfiguration('jpeg_quality')
    scaler_crop_str = context.perform_substitution(LaunchConfiguration('scaler_crop'))

    camera_parameters = {
        'width': width,
        'height': height,
        'jpeg_quality': jpeg_quality,
        'orientation': LaunchConfiguration('orientation'),
    }
    if scaler_crop_str.strip():
        crop_values = [int(v.strip()) for v in scaler_crop_str.split(',')]
        if len(crop_values) != 4:
            raise ValueError(
                f"scaler_crop must be 'x,y,w,h' (4 integers), got {scaler_crop_str!r}"
            )
        camera_parameters['ScalerCrop'] = crop_values

    camera_container = ComposableNodeContainer(
        name='camera_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='camera_ros',
                plugin='camera::CameraNode',
                name='camera',
                parameters=[camera_parameters],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
        output='screen',
    )
    return [camera_container]


def generate_launch_description():
    usb_port = LaunchConfiguration('usb_port')
    tb3_param_dir = LaunchConfiguration('tb3_param_dir')
    vision_debug = LaunchConfiguration('vision_debug')
    vision_debug_config_file = LaunchConfiguration('vision_debug_config_file')

    turtlebot3_bringup_share = get_package_share_directory('turtlebot3_bringup')
    this_pkg = get_package_share_directory('simple_camera_pid')

    # namespace is a required (but unset-by-default) parameter on this
    # turtlebot3_node build -- an empty string can't be passed via
    # -p namespace:="" (the ROS2 arg parser rejects empty-string overrides),
    # so it's set the same way real_line_follower.launch.py overrides
    # individual params: an inline dict layered after the main param file.
    base_driver = Node(
        package='turtlebot3_node',
        executable='turtlebot3_ros',
        output='screen',
        arguments=['-i', usb_port],
        parameters=[
            tb3_param_dir,
            {'namespace': ''},
        ],
    )

    vision_debug_node = Node(
        package='simple_camera_pid',
        executable='vision_debug_node',
        name='vision_debug_node',
        output='screen',
        parameters=[
            vision_debug_config_file,
            {'platform': 'real'},
        ],
        condition=IfCondition(vision_debug),
    )

    return LaunchDescription([
        OpaqueFunction(function=_warn_if_env_mismatch),
        DeclareLaunchArgument(
            'width', default_value='640',
            description='Camera capture width in pixels.',
        ),
        DeclareLaunchArgument(
            'height', default_value='480',
            description='Camera capture height in pixels.',
        ),
        DeclareLaunchArgument(
            'jpeg_quality', default_value='70',
            description='/camera/image_raw/compressed JPEG quality (0-100).',
        ),
        DeclareLaunchArgument(
            'usb_port', default_value='/dev/ttyACM0',
            description='OpenCR serial port for the base driver.',
        ),
        DeclareLaunchArgument(
            'tb3_param_dir',
            default_value=os.path.join(
                turtlebot3_bringup_share, 'param', 'burger.yaml'),
            description='TurtleBot3 kinematics/motor parameter YAML.',
        ),
        DeclareLaunchArgument(
            'vision_debug', default_value='true',
            description='Launch vision_debug_node (must run on this robot, '
                        'not the PC -- see module docstring).',
        ),
        DeclareLaunchArgument(
            'vision_debug_config_file',
            default_value=os.path.join(
                this_pkg, 'config', 'real', 'real_line_follower.yaml'),
            description='Vision/camera tuning YAML for vision_debug_node -- '
                        'same file the real controller nodes read, so its '
                        'view can never drift from what the robot runs.',
        ),
        DeclareLaunchArgument(
            'orientation', default_value='180',
            description='camera_ros/libcamera "orientation" control (0/90/180/270 '
                        'degrees, read-only -- set at node startup, cannot change '
                        'live). Default 180 corrects for THIS robot\'s camera '
                        'being mounted physically upside-down (confirmed '
                        '2026-08-07) -- rotates the sensor readout before any '
                        'output scaling/cropping, so width/height/scaler_crop '
                        'above are unaffected. Pass orientation:=0 if the mount '
                        'is ever corrected/reversed.',
        ),
        DeclareLaunchArgument(
            'scaler_crop', default_value='',
            description='libcamera ScalerCrop "x,y,w,h" in full-sensor '
                        'pixel space. Empty (default) = unset = uncropped, '
                        'current behavior. UNCALIBRATED -- see module '
                        'docstring before using a non-empty value.',
        ),
        OpaqueFunction(function=_make_camera_container),
        base_driver,
        vision_debug_node,
    ])
