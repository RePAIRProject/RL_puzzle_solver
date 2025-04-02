import os 
import pandas as pd 
import json



methods = ['deeplsd', 'noise_positional_p30_s1', 'noise_positional_p30_s2', 'noise_positional_p30_s3', 'noise_structural_p5', 'noise_structural_p10']

root_folder = 'output_8x8/random_lines_exact_detection'

methods_perf = {
    "deeplsd": {
        'simple': [],
        'vector': [],
        'neighbours': [],
        'pixel': []
    },
    "noise_positional_p30_s1": {
        'simple': [],
        'vector': [],
        'neighbours': [],
        'pixel': []
    },
    "noise_positional_p30_s2": {
        'simple': [],
        'vector': [],
        'neighbours': [],
        'pixel': []
    },
    "noise_positional_p30_s3": {
        'simple': [],
        'vector': [],
        'neighbours': [],
        'pixel': []
    },
    "noise_structural_p5": {
        'simple': [],
        'vector': [],
        'neighbours': [],
        'pixel': []
    },
    "noise_structural_p10": {
        'simple': [],
        'vector': [],
        'neighbours': [],
        'pixel': []
    }
}

counter = {}
for method in methods:
    counter[method] = 0

for n in range(0, 30):
    eval_fld = os.path.join(root_folder, f"image_{n}", 'evaluation')
    for method in methods:
        jpath = os.path.join(eval_fld, method, 'quantitative_evaluation', f'evaluation_{method}.json')
        if os.path.exists(jpath):
            counter[method] += 1
            with open(jpath, 'r') as jp:
                perf_cur_img = json.load(jp)
            methods_perf[method]['simple'].append(perf_cur_img['correct'])
            methods_perf[method]['vector'].append(perf_cur_img['correct_vector'])
            methods_perf[method]['neighbours'].append(perf_cur_img['neighbours'])
            methods_perf[method]['pixel'].append(perf_cur_img['pixel'])
        else:
            methods_perf[method]['simple'].append(0)
            methods_perf[method]['vector'].append(0)
            methods_perf[method]['neighbours'].append(0)
            methods_perf[method]['pixel'].append(0)

noisy_performances_df = pd.DataFrame()
noisy_performances_df['deeplsd_simple'] = methods_perf['deeplsd']['simple']
noisy_performances_df['deeplsd_vector'] = methods_perf['deeplsd']['vector']
noisy_performances_df['deeplsd_neighbours'] = methods_perf['deeplsd']['neighbours']
noisy_performances_df['deeplsd_pixel'] = methods_perf['deeplsd']['pixel']

noisy_performances_df['noise_positional_p30_s1_simple'] = methods_perf['noise_positional_p30_s1']['simple']
noisy_performances_df['noise_positional_p30_s1_vector'] = methods_perf['noise_positional_p30_s1']['vector']
noisy_performances_df['noise_positional_p30_s1_neighbours'] = methods_perf['noise_positional_p30_s1']['neighbours']
noisy_performances_df['noise_positional_p30_s1_pixel'] = methods_perf['noise_positional_p30_s1']['pixel']

noisy_performances_df['noise_positional_p30_s2_simple'] = methods_perf['noise_positional_p30_s2']['simple']
noisy_performances_df['noise_positional_p30_s2_vector'] = methods_perf['noise_positional_p30_s2']['vector']
noisy_performances_df['noise_positional_p30_s2_neighbours'] = methods_perf['noise_positional_p30_s2']['neighbours']
noisy_performances_df['noise_positional_p30_s2_pixel'] = methods_perf['noise_positional_p30_s2']['pixel']

noisy_performances_df['noise_positional_p30_s3_simple'] = methods_perf['noise_positional_p30_s3']['simple']
noisy_performances_df['noise_positional_p30_s3_vector'] = methods_perf['noise_positional_p30_s3']['vector']
noisy_performances_df['noise_positional_p30_s3_neighbours'] = methods_perf['noise_positional_p30_s3']['neighbours']
noisy_performances_df['noise_positional_p30_s3_pixel'] = methods_perf['noise_positional_p30_s3']['pixel']

noisy_performances_df['noise_structural_p5_simple'] = methods_perf['noise_structural_p5']['simple']
noisy_performances_df['noise_structural_p5_vector'] = methods_perf['noise_structural_p5']['vector']
noisy_performances_df['noise_structural_p5_neighbours'] = methods_perf['noise_structural_p5']['neighbours']
noisy_performances_df['noise_structural_p5_pixel'] = methods_perf['noise_structural_p5']['pixel']

noisy_performances_df['noise_structural_p10_simple'] = methods_perf['noise_structural_p10']['simple']
noisy_performances_df['noise_structural_p10_vector'] = methods_perf['noise_structural_p10']['vector']
noisy_performances_df['noise_structural_p10_neighbours'] = methods_perf['noise_structural_p10']['neighbours']
noisy_performances_df['noise_structural_p10_pixel'] = methods_perf['noise_structural_p10']['pixel']

noisy_performances_df.to_csv(os.path.join(root_folder, 'evaluation_noisy_v2.csv'))

print("#" * 40)
print("DEEPLSD")
print(f"simple: {(noisy_performances_df.loc[:, 'deeplsd_simple'].sum()/counter['deeplsd']):.03f}")
print(f"vector: {(noisy_performances_df.loc[:, 'deeplsd_vector'].sum()/counter['deeplsd']):.02f}")
print(f"neighbours: {(noisy_performances_df.loc[:, 'deeplsd_neighbours'].sum()/counter['deeplsd']):.03f}")
print(f"pixel error: {(noisy_performances_df.loc[:, 'deeplsd_pixel'].sum()/counter['deeplsd']):.03f}")
print(f"pixel acc: { (1 - (noisy_performances_df.loc[:, 'deeplsd_pixel'].sum()/counter['deeplsd'])):.03f}")
print(f"evaluated on {counter['deeplsd']} images")

print("-" * 40)
print("NOISE p30 s1")
print(f"simple: {(noisy_performances_df.loc[:, 'noise_positional_p30_s1_simple'].sum()/counter['noise_positional_p30_s1']):.03f}")
print(f"vector: {(noisy_performances_df.loc[:, 'noise_positional_p30_s1_vector'].sum()/counter['noise_positional_p30_s1']):.02f}")
print(f"neighbours: {(noisy_performances_df.loc[:, 'noise_positional_p30_s1_neighbours'].sum()/counter['noise_positional_p30_s1']):.03f}")
print(f"pixel error: {(noisy_performances_df.loc[:, 'noise_positional_p30_s1_pixel'].sum()/counter['noise_positional_p30_s1']):.03f}")
print(f"pixel acc: { (1 - (noisy_performances_df.loc[:, 'noise_positional_p30_s1_pixel'].sum()/counter['noise_positional_p30_s1'])):.03f}")
print(f"evaluated on {counter['noise_positional_p30_s1']} images")

print("-" * 40)
print("NOISE p30 s2")
print(f"simple: {(noisy_performances_df.loc[:, 'noise_positional_p30_s2_simple'].sum()/counter['noise_positional_p30_s2']):.03f}")
print(f"vector: {(noisy_performances_df.loc[:, 'noise_positional_p30_s2_vector'].sum()/counter['noise_positional_p30_s2']):.02f}")
print(f"neighbours: {(noisy_performances_df.loc[:, 'noise_positional_p30_s2_neighbours'].sum()/counter['noise_positional_p30_s2']):.03f}")
print(f"pixel error: {(noisy_performances_df.loc[:, 'noise_positional_p30_s2_pixel'].sum()/counter['noise_positional_p30_s2']):.03f}")
print(f"pixel acc: { (1 - (noisy_performances_df.loc[:, 'noise_positional_p30_s2_pixel'].sum()/counter['noise_positional_p30_s2'])):.03f}")
print(f"evaluated on {counter['noise_positional_p30_s2']} images")

print("-" * 40)
print("NOISE p30 s3")
print(f"simple: {(noisy_performances_df.loc[:, 'noise_positional_p30_s3_simple'].sum()/counter['noise_positional_p30_s3']):.03f}")
print(f"vector: {(noisy_performances_df.loc[:, 'noise_positional_p30_s3_vector'].sum()/counter['noise_positional_p30_s3']):.02f}")
print(f"neighbours: {(noisy_performances_df.loc[:, 'noise_positional_p30_s3_neighbours'].sum()/counter['noise_positional_p30_s3']):.03f}")
print(f"pixel error: {(noisy_performances_df.loc[:, 'noise_positional_p30_s3_pixel'].sum()/counter['noise_positional_p30_s3']):.03f}")
print(f"pixel acc: { (1 - (noisy_performances_df.loc[:, 'noise_positional_p30_s3_pixel'].sum()/counter['noise_positional_p30_s3'])):.03f}")
print(f"evaluated on {counter['noise_positional_p30_s3']} images")

print("-" * 40)
print("MISSING p5")
print(f"simple: {(noisy_performances_df.loc[:, 'noise_structural_p5_simple'].sum()/counter['noise_structural_p5']):.03f}")
print(f"vector: {(noisy_performances_df.loc[:, 'noise_structural_p5_vector'].sum()/counter['noise_structural_p5']):.02f}")
print(f"neighbours: {(noisy_performances_df.loc[:, 'noise_structural_p5_neighbours'].sum()/counter['noise_structural_p5']):.03f}")
print(f"pixel error: {(noisy_performances_df.loc[:, 'noise_structural_p5_pixel'].sum()/counter['noise_structural_p5']):.03f}")
print(f"pixel acc: { (1 - (noisy_performances_df.loc[:, 'noise_structural_p5_pixel'].sum()/counter['noise_structural_p5'])):.03f}")
print(f"evaluated on {counter['noise_structural_p5']} images")

print("-" * 40)
print("MISSING p10")
print(f"simple: {(noisy_performances_df.loc[:, 'noise_structural_p10_simple'].sum()/counter['noise_structural_p10']):.03f}")
print(f"vector: {(noisy_performances_df.loc[:, 'noise_structural_p10_vector'].sum()/counter['noise_structural_p10']):.02f}")
print(f"neighbours: {(noisy_performances_df.loc[:, 'noise_structural_p10_neighbours'].sum()/counter['noise_structural_p10']):.03f}")
print(f"pixel error: {(noisy_performances_df.loc[:, 'noise_structural_p10_pixel'].sum()/counter['noise_structural_p10']):.03f}")
print(f"pixel acc: { (1 - (noisy_performances_df.loc[:, 'noise_structural_p10_pixel'].sum()/counter['noise_structural_p10'])):.03f}")
print(f"evaluated on {counter['noise_structural_p10']} images")
print("#" * 40)