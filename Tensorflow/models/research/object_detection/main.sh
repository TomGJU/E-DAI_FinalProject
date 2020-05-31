while :
do
    python3 model_main.py --pipeline_config_path=/home/tom/TensorFlow/models/research/object_detection/export/pipeline.config --model_dir=/home/tom/TensorFlow/models/research/object_detection/export/checkpoint --sample_1_of_n_eval_examples=1 --alsologtostderr
    python3 recup.py 
done