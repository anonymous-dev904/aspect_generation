import subprocess
import os
root = os.path.dirname(__file__)


# generation
name_list = ['vul', 'root', 'vec', 'impact']
for one_name in name_list[2:3]:

    train_file = root + '/output/github_cwe_code_class_train_codebert_mix_synk_nvd_generation_allhave_' + one_name + '_clean_seg_cut.jsonl'
    test_file = root + '/output/github_cwe_code_class_test_codebert_mix_synk_nvd_generation_allhave_' + one_name + '_clean_seg_cut.jsonl'
    output_dir = root + '/output/github_cwe_code_class_codebert_mix_synk_nvd_generation_allhave_' + one_name + '_clean_seg_cut_test_layer/'
    print(train_file)
    print(test_file)
    print(output_dir)
    try:
        print(subprocess.check_output(
            [r"D:\PyCharmProj\BART_abstract\venv\Scripts\python.exe",
             r"D:\PyCharmProj\BART_abstract\code2nl_cross\run.py",
             "--train_filename", train_file, "--test_filename",
             test_file, "--output_dir", output_dir,]))# "--load_model_path", output_dir + 'checkpoint-best-ppl/pytorch_model.bin'
    except Exception as e:
        print(e)

    # train_file = root + '/output/github_cwe_code_class_train_codebert_mix_synk_nvd_generation_allhave_' + one_name + '_clean_seg_cut.jsonl'
    # test_file = root + '/output/github_cwe_code_class_test_codebert_mix_synk_nvd_generation_allhave_' + one_name + '_clean_seg_cut.jsonl'
    # output_dir = root + '/output/github_cwe_code_class_codebert_mix_synk_nvd_generation_allhave_' + one_name + '_clean_seg_cut10_layer/'
    # print(train_file)
    # print(test_file)
    # print(output_dir)
    # try:
    #     print(subprocess.check_output(
    #         [r"D:\PyCharmProj\BART_abstract\venv\Scripts\python.exe",
    #          r"D:\PyCharmProj\BART_abstract\code2nl_cross\run.py",
    #          "--train_filename", train_file, "--test_filename",
    #          test_file, "--output_dir",
    #          output_dir, "--num_train_epochs", "10"]))  # "--load_model_path", output_dir + 'checkpoint-best-ppl/pytorch_model.bin'
    # except Exception as e:
    #     print(e)