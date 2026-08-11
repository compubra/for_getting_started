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

``line_follower_vision_node`` (``vision_node:=true``, default false) is the
other opt-in node this file can start -- the split-deployment vision half,
publishing ``/line_follower/local_path`` for a PC-side
``line_follower_control_node`` to consume (see that node's own module
docstring for the split architecture's rationale). It reads
``vision_node_config_file`` (same default yaml as ``vision_debug_config_file``
below). Turn ``vision_debug`` off when turning this on -- see ``vision_node``'s
own argument description for why.

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
    ros2 launch simple_camera_pid robot_bringup.launch.py \\
        vision_debug:=false vision_node:=true
        (split-deployment control tuning: this robot only runs the camera,
        base driver, and line_follower_vision_node -- vision_debug_node stays
        off so it isn't competing for CPU. Pair with a PC-side
        line_follower_control_node, e.g. via ~/.bashrc's pi_vision, which now
        just wraps this same argument.)
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
    vision_node              Launch line_follower_vision_node, the split-
                          deployment vision half (default false -- see its
                          own entry below).
    vision_node_config_file
                            Vision/camera tuning YAML for
                          line_follower_vision_node (default same
                          real_line_follower.yaml).
    auto_exposure           Leave libcamera AE/AGC on (default false). The
                          default LOCKS exposure -- this is what replaces
                          "turn the lab lights off" as the way to stop the
                          camera re-metering mid-lap. See the argument's own
                          description below for the measurements.
    exposure_time           Manual ExposureTime in microseconds (default
                          49000). Ignored when auto_exposure:=true.
    analogue_gain           Manual AnalogueGain (default 10.0). Ignored when
                          auto_exposure:=true. AWB stays on deliberately.
    frame_duration          FrameDurationLimits in microseconds, min=max
                          (default 50000 = 20 fps), which is what lets
                          exposure_time exceed the 33333 us ceiling 30 fps
                          imposes.
                          THE THREE ABOVE DEFAULT TO THE LIGHTS-OFF ROOM
                          measured 2026-08-10. With the room lights ON use
                          exposure_time:=24000 analogue_gain:=2.0
                          frame_duration:=33333 AND set min_brightness back to
                          170 in real_line_follower.yaml -- the exposure and
                          the mask floor are one calibration, see that file.
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

    # Exposure lock. See auto_exposure's argument description below for why
    # this is not optional for line following, and how to re-measure the two
    # numbers. Both mode controls must be set: on libcamera 0.5.2 (this Pi)
    # ExposureTimeMode/AnalogueGainMode take precedence over AeEnable in
    # camera_ros's ParameterConflictHandler, so setting AeEnable:=false alone
    # leaves ExposureTime rejected with "must not be set simultaneously" and
    # the camera silently stays on auto. 1 == Manual for both.
    # These have to be passed at node construction, not via ros2 param set
    # afterwards: camera_ros applies queued control values on the next
    # capture request and drops any that arrive before the previous set has
    # been consumed ("previous parameters have note been apllied yet and
    # will be ignored"), which makes runtime setting unreliable.
    if context.perform_substitution(LaunchConfiguration('auto_exposure')).lower() \
            not in ('true', '1'):
        camera_parameters['ExposureTimeMode'] = 1
        camera_parameters['AnalogueGainMode'] = 1
        camera_parameters['ExposureTime'] = LaunchConfiguration('exposure_time')
        camera_parameters['AnalogueGain'] = LaunchConfiguration('analogue_gain')

    # FrameDurationLimits caps how long ExposureTime is allowed to be: at the
    # default 30 fps the ceiling is 33333 us, and in a dim room that forces
    # the shortfall onto AnalogueGain. That is the worse trade -- measured in
    # the dark 2026-08-10, 33000 us x8.0 gave a line of only 93 with
    # saturation p90 0.213 (past max_saturation 0.20), because saturation
    # here is dominated by how dark the pixel is, not by gain as such. The
    # control loop only consumes 20 Hz (sample-and-hold in
    # line_follower_node), so trading frame rate for exposure time is free
    # down to 20 fps. Set both limits equal to pin the rate exactly.
    frame_duration_str = context.perform_substitution(
        LaunchConfiguration('frame_duration')).strip()
    if frame_duration_str:
        fd = int(frame_duration_str)
        camera_parameters['FrameDurationLimits'] = [fd, fd]

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
    vision_node_config_file = LaunchConfiguration('vision_node_config_file')

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

    # Split-deployment vision half (see vision_node.py's module docstring and
    # ~/.bashrc's pi_vision -- this is what that function used to start
    # separately). Off by default: it only makes sense when a PC-side
    # line_follower_control_node is also running, and running it alongside
    # vision_debug_node would subscribe to the raw camera feed twice on the
    # already CPU-constrained Pi for no benefit. camera_profile defaults to
    # "real" inside the node itself, so no override dict is needed here the
    # way vision_debug_node needs {'platform': 'real'}.
    vision_node = Node(
        package='simple_camera_pid',
        executable='line_follower_vision_node',
        name='line_follower_vision_node',
        output='screen',
        parameters=[vision_node_config_file],
        condition=IfCondition(LaunchConfiguration('vision_node')),
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
            'vision_node', default_value='false',
            description='Launch line_follower_vision_node -- the split-'
                        'deployment vision half, publishing '
                        '/line_follower/local_path for a PC-side '
                        'line_follower_control_node to consume. Default '
                        'false: only turn this on when actually running the '
                        'split control-tuning setup, and pair it with '
                        'vision_debug:=false (both subscribe to the raw '
                        'camera feed; running both at once competes for the '
                        "Pi's CPU for no reason -- see vision_debug_node's "
                        'own module docstring for the same argument applied '
                        'there).',
        ),
        DeclareLaunchArgument(
            'vision_node_config_file',
            default_value=os.path.join(
                this_pkg, 'config', 'real', 'real_line_follower.yaml'),
            description='Vision/camera tuning YAML for line_follower_vision_node '
                        '-- same file vision_debug_config_file points at by '
                        'default (and the same the PC-side control_node '
                        'reads its own copy of), kept as a separate argument '
                        'in case you ever want the debug overlay and the '
                        'real detector to run against two different files.',
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
            'auto_exposure', default_value='false',
            description='Leave libcamera auto-exposure/auto-gain on. Default '
                        'false -- AE is actively harmful for this line '
                        'follower and locking it is what removes the need to '
                        'run the lab with the lights off. The scene is a '
                        'near-black carpet with a white tape line, so AE '
                        'meters almost entirely on how much line is currently '
                        'in view: with the line filling the near field it '
                        'stops down, and as the line leaves the frame it opens '
                        'up and lifts the bare carpet into the same brightness '
                        'band as the line. vision.py then compounds it, '
                        'because its mask cut used to be proportional to the '
                        "ROI's brightest pixel (see "
                        'LineFollowerVision._brightness_threshold). Measured '
                        'over three 2026-08-06/07 bags: the applied cut ranged '
                        '40-89, the mask covered 23-48% of the ROI on 10-32% '
                        'of frames, and the white line itself never read above '
                        '162. Set true only to reproduce that old behaviour.',
        ),
        DeclareLaunchArgument(
            'exposure_time', default_value='49000',
            description='Manual ExposureTime in microseconds, used unless '
                        'auto_exposure:=true. 24000 measured 2026-08-10 on '
                        'this robot, stationary and centred on the track with '
                        'the lab lights ON: line value p10 201 / p50 207, '
                        'ground p99 114 / max 126, no clipped pixels, line '
                        'saturation p90 0.048 (well inside max_saturation '
                        '0.20). That leaves min_brightness: 170 sitting '
                        'roughly midway in the gap, ~31 counts clear of both '
                        'sides. NOT a whole-track calibration -- it is one '
                        'pose under one lighting condition. Re-measure if the '
                        'lighting, the track surface, or the camera changes: '
                        'point the robot at the darkest and the brightest '
                        'stretch of the lap and check the line still reads '
                        'above ~190 and the carpet below ~140 at both. '
                        'Exposure and gain trade off linearly and exactly '
                        '(4000us x2.0 measured identical to 8000us x1.0), so '
                        'shorten this and raise analogue_gain if motion blur '
                        'ever matters -- at 24 ms and 0.06 m/s the smear is '
                        '~1.4 mm, negligible against a 130-161 px line, but '
                        'that scales with speed. Keep it under the frame '
                        'period (33333 us at 30 fps) or the frame rate drops.',
        ),
        DeclareLaunchArgument(
            'analogue_gain', default_value='10.0',
            description='Manual AnalogueGain, used unless auto_exposure:=true. '
                        'See exposure_time above -- 2.0 is the gain half of '
                        'that same 2026-08-10 measurement. AWB is deliberately '
                        'left ON: it is not part of the problem (auto-exposure '
                        'is), and switching it off without also supplying '
                        'calibrated ColourGains puts a heavy colour cast on '
                        'the frame that pushes the white line to saturation '
                        'p90 ~0.45, past the max_saturation 0.20 gate, so the '
                        'line stops being detected at all. Measured: AWB on '
                        'with exposure locked gives line saturation p90 '
                        '0.048-0.090.',
        ),
        DeclareLaunchArgument(
            'frame_duration', default_value='50000',
            description='libcamera FrameDurationLimits, in microseconds, set '
                        'as both the min and max so the frame rate is pinned '
                        'exactly. Empty (default) = leave it alone, i.e. 30 fps '
                        'and a 33333 us ceiling on exposure_time. Set 50000 '
                        '(20 fps) in a dim room so exposure_time can go past '
                        'that ceiling instead of pushing AnalogueGain -- see '
                        'the comment at the assignment. Do not go below 20 fps: '
                        "line_follower_node's control timer runs at 20 Hz and "
                        'would start re-using frames.',
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
        vision_node,
    ])
