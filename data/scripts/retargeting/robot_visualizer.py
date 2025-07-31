import mujoco
import mujoco.viewer  # for visualization

# Path to your robot MJCF file
xml_path = "protomotions/data/assets/mjcf/male_human_model.xml"
# xml_path = "protomotions/data/assets/mjcf/g1.xml"
# xml_path = "robot.xml"

# Load model from XML file
model = mujoco.MjModel.from_xml_path(xml_path)

# Create a simulation data object
data = mujoco.MjData(model)

# (Optional) Open an interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()


