project_dir=/root/KarmaLoop
cd ${project_dir}/utils
# start
for task_job in single_example CASP1314 CASP15 ab_benchmark 
do
echo "### ${task_job}"
logf=${project_dir}/example/${task_job}/${task_job}.log
csv_file=${project_dir}/example/${task_job}/${task_job}.csv
protein_fir_dir=${project_dir}/example/${task_job}/raw
pocket_file_dir=${project_dir}/example/${task_job}/pocket
graph_file_dir=${project_dir}/example/${task_job}/graph
# preprocessing
echo "### ${task_job} preprocessing"
python -u pre_processing.py --protein_file_dir ${protein_fir_dir} --pocket_file_dir ${pocket_file_dir} --csv_file ${csv_file} > ${logf}
# generating graph
echo "### ${task_job} generating graph"
python -u generate_graph.py --protein_file_dir ${protein_fir_dir} --pocket_file_dir ${pocket_file_dir} --graph_file_dir ${graph_file_dir} >> ${logf}
# loop generating SINGLE-CONFORMATION
echo "### ${task_job} loop generating SINGLE-CONFORMATION"
python -u loop_generating.py --graph_file_dir ${graph_file_dir} --out_dir ${project_dir}/example/${task_job}/singleconf_result --batch_size 64 --random_seed 2023 --scoring --save_file >> ${logf}
# loop generating MULTI-CONFORMATION
echo "### ${task_job} loop generating MULTI-CONFORMATION"
python -u loop_generating.py --graph_file_dir ${graph_file_dir} --out_dir ${project_dir}/example/${task_job}/multiconf_result --batch_size 64 --random_seed 2023 --multi_conformation --scoring --save_file >> ${logf}
## openmm minimization (optional)
# echo "### ${task_job} openmm minimization"
# python -u post_processing.py --modeled_loop_dir ${project_dir}/example/${task_job}/singleconf_result/0 --output_dir ${project_dir}/example/${task_job}/singleconf_result/0/post >> ${logf}
done