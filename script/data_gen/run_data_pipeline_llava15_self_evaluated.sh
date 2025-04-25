# An example of LLaVA 1.5 7B generate - divide and conquer - OmniLMM 12B feedback

## step1: diverse gen
model_path=models/llava-v1.5-7b # path of the model
ques_dir=./cot_one # directory of the input file
ques_file=viscot_363k_rlhf_30 # name of the input file, without '.jsonl'
ans_dir=./rlhf_result_30 # directory of the answer files
start=0
end=-1
# bash ./script/data_gen/llava15/llava15_diverse_gen.sh \
# $model_path \
# $ans_dir \
# $ques_dir \
# ${ques_file}.jsonl \
# $start \
# $end \
# 8 # num of gpus that can be used

# python ./script/data_gen/generate_next_question.py --input_file ${ans_dir}/diverse_gen_llava15_${start}-${end}_${ques_file}.jsonl --output_file ${ans_dir}/${ques_file}_next.jsonl

# bash ./script/data_gen/llava15/llava15_diverse_gen_next.sh \
# $model_path \
# $ans_dir \
# $ques_dir \
# ${ques_file}_next.jsonl \
# $start \
# $end \
# 8 # num of gpus that can be used



# step2: llama3 divide
ans_file=${ans_dir}/diverse_gen_llava15_${start}-${end}_${ques_file} # path of the generation result file
# bash ./script/data_gen/divide_and_conquer/llama3_8b_divide_and_conquer.sh \
# $ans_file \
# 0 \
# -1 \
# 8 \
# 8 # chunk num, equal to the gpus available

## step3: autocheck
model_path=../Visual-CoT/checkpoints/llava-VisCoT-7b-336
check_ques_file=diverse_gen_llava15_${start}-${end}_${ques_file}_next.s0-e-1.llama3-8b_divide.gq.qas.jsonl # name of the divide-and-conquer result file

# bash ./script/data_gen/llava15/self_evaluate.sh \
# $model_path \
# $ans_dir \
# $ans_dir \
# $check_ques_file \
# 0 \
# -1 \
# 8 # num of gpus that can be used

# step4: construct preference pairs
gq_file=diverse_gen_llava15_${start}-${end}_${ques_file}_next.s0-e-1.llama3-8b_divide.gq.jsonl # name of the divided answer file
feedback_file=autocheck_llava_0--1_diverse_gen_llava15_${start}-${end}_${ques_file}_next.s0-e-1.llama3-8b_divide.gq.qas.jsonl # name of the feedback result file
# bash ./script/data_gen/construct_pairs.sh \
# ${ans_dir}/${feedback_file} \
# ${ans_dir}/${gq_file} \
# 2 # num of the sample pairs for each instruction

# balance win - lose answer length
pairs_file=autocheck_llava_0--1_diverse_gen_llava15_${start}-${end}_${ques_file}_next.s0-e-1.llama3-8b_divide.gq.qas_pair_diff1_samp2.jsonl
balanced_pairs_file=autocheck_llava_0--1_diverse_gen_llava15_${start}-${end}_${ques_file}_next.s0-e-1.llama3-8b_divide.gq.qas_pair_diff1_samp2_balanceshort.jsonl
python ./utils/get_pairs_filter_shorten.py \
--path ${ans_dir}/$pairs_file \
--save_path ${ans_dir}/$balanced_pairs_file