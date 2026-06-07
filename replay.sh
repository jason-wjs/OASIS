SCRIPT_DIR=$(dirname $(realpath $0))

input_dir="${SCRIPT_DIR}/tasks/teleop/record_data/0509_pick_up_basket"
output_dir="${SCRIPT_DIR}/tasks/teleop/aug_data/0509_pick_up_basket"
cd tasks/teleop

python server_low_level_g1_isaacsim.py \
    --device cuda \
    --enable_cameras \
    --replay \
    --input_dir ${input_dir} \
    --output_dir ${output_dir} \
    --controller "teleopit" \
    --num_envs 1 \
    --start 0 \
    --end 50 \
    --target_envs_per_episode 20 \
    #--headless



