# Brain_column_analyse

This repository is a backup for codes in brain column analyse.

## Instruction of usage

To do columns analyses, run the following scripts in sequence.

- `submit_job_alex.sh`: submit all jobs in `input_dir` and put output to `output_dir`

- `bbregister_hanwen.sh`: generate transformation matrix
	- `--s`: the name of subject, must align with the `submit_job_alex.sh`
	- `--mov`: the move image
	- `--reg`: the stationary image

- `vertices_connect.m`: generate paired vertices points in anatomical and dwi domain
	- change the `ID` and `output_dir` if needed
	- change the `trans_M_dir` of transformation matrix dictionary if needed

- `view_columns.m`: view the difussion data along columns as needed
