import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def wilcox_2_samp(
    feature_frame,
    label_series,
    group_1_name='pCR',
    group_2_name='pNR',
    alt_hyp='two-sided',
    multi_method=None
):

    # options for the alternative hypothesis are two-sided, less, and greater

    # grab the unique stat names
    col_names = feature_frame.columns.tolist()

    # get positive and negative sample names
    pos_samples = label_series[label_series==group_1_name].index.tolist()
    neg_samples = label_series[label_series==group_2_name].index.tolist()

    results_dict = {}
    for col_name in col_names:
        stat_dict = {}
        stat_series = feature_frame[col_name]

        pos_scores = stat_series.loc[pos_samples].values
        neg_scores = stat_series.loc[neg_samples].values

        U, p = stats.mannwhitneyu(pos_scores, neg_scores, alternative=alt_hyp, method='auto')

        U_upper = (len(pos_scores)*len(neg_scores))-U

        auc_score_upper = U_upper/(len(pos_scores)*len(neg_scores))
        auc_score_lower = U/(len(pos_scores)*len(neg_scores))

        if auc_score_upper>auc_score_lower:
            auc_score = auc_score_upper
        elif auc_score_lower>auc_score_upper:
            auc_score = auc_score_lower
        else:
            auc_score = auc_score_lower

        stat_dict['U_stat'] = U
        stat_dict['U_upper'] = U_upper
        stat_dict['U_lower'] = U
        stat_dict['AUC'] = auc_score
        stat_dict['AUC_upper'] = auc_score_upper
        stat_dict['AUC_lower'] = auc_score_lower
        stat_dict['p_val'] = p

        results_dict[col_name] = stat_dict

    if multi_method is not None:
        # grab all initial pvals
        p_val_list = [results_dict[stat_name]['p_val'] for stat_name in col_names]

        reject, corrected_p_vals, _, _ = multipletests(p_val_list, alpha=0.05, method=multi_method, is_sorted=False, returnsorted=False)

        # create and save a new dictionary with the created p-vals
        corrected_stat_dict = {}
        for col_index, col_name in enumerate(col_names):
            new_stat_dict = {}
            old_stat_dict = results_dict[col_name]
            new_stat_dict['U_stat'] = old_stat_dict['U_stat']
            new_stat_dict['AUC'] = old_stat_dict['AUC']
            new_stat_dict['p_val'] = corrected_p_vals[col_index]
            new_stat_dict['reject'] = reject[col_index]
            new_stat_dict['U_upper'] = old_stat_dict['U_upper']
            new_stat_dict['U_lower'] = old_stat_dict['U_lower']
            new_stat_dict['AUC_upper'] = old_stat_dict['AUC_upper']
            new_stat_dict['AUC_lower'] = old_stat_dict['AUC_lower']


            corrected_stat_dict[col_name] = new_stat_dict

    else:
        corrected_stat_dict = results_dict

    return pd.DataFrame(corrected_stat_dict).T

def ttest_2_samp(
    feature_frame,
    label_series,
    group_1_name='pCR',
    group_2_name='pNR',
    alt_hyp='two-sided',
    multi_method=None
):
    # Supported alternative hypotheses: 'two-sided', 'less', 'greater'
    
    col_names = feature_frame.columns.tolist()
    pos_samples = label_series[label_series == group_1_name].index.tolist()
    neg_samples = label_series[label_series == group_2_name].index.tolist()

    results_dict = {}
    for col_name in col_names:
        stat_dict = {}
        stat_series = feature_frame[col_name]

        pos_scores = stat_series.loc[pos_samples].values
        neg_scores = stat_series.loc[neg_samples].values

        t_stat, p_val = stats.ttest_ind(pos_scores, neg_scores, alternative=alt_hyp, equal_var=False)

        stat_dict['t_stat'] = t_stat
        stat_dict['p_val'] = p_val

        results_dict[col_name] = stat_dict

    if multi_method is not None:
        p_val_list = [results_dict[stat_name]['p_val'] for stat_name in col_names]
        reject, corrected_p_vals, _, _ = multipletests(p_val_list, alpha=0.05, method=multi_method, is_sorted=False, returnsorted=False)

        corrected_stat_dict = {}
        for col_index, col_name in enumerate(col_names):
            new_stat_dict = {
                't_stat': results_dict[col_name]['t_stat'],
                'p_val': corrected_p_vals[col_index],
                'reject': reject[col_index]
            }
            corrected_stat_dict[col_name] = new_stat_dict
    else:
        corrected_stat_dict = results_dict

    return pd.DataFrame(corrected_stat_dict).T


