# sudo ufw disable

source ~/miniconda3/bin/activate gmr

cd tasks/teleop


redis_ip="localhost"

# the height (empirically) should be smaller than the actual human height, due to inaccuracy of the PICO estimation.
actual_human_height=1.6
python teleop_retarget.py --robot unitree_g1 \
             --actual_human_height $actual_human_height \
             --redis_ip $redis_ip \
             --target_fps 50 \
             --measure_fps 1 \


