SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/teleopit/track.onnx
record_dir="${SCRIPT_DIR}/tasks/teleop/record_data"
redis_ip="localhost"
cd tasks/teleop

python server.py \
    --policy ${ckpt_path} \
    --device cpu \
    --dt 0.005 \
    --decimation 4 \
    --redis_ip $redis_ip \
    --record \
    --record_dir ${record_dir} \
    --enable_cameras \
    --controller "teleopit" \
    --scene "table_white_basket" \
    --text "pick up the basket" \
    --pico_host 192.168.1.1 \
    --headless
