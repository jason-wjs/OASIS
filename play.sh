SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/teleopit/track.onnx
redis_ip="localhost"
cd tasks/teleop

python server.py \
    --policy ${ckpt_path} \
    --device cpu \
    --dt 0.005 \
    --decimation 4 \
    --redis_ip $redis_ip \
    --scene "table_white_basket" \
    --controller "teleopit" \
    --enable_cameras \
    --pico_host 192.168.1.1
    #--headless

