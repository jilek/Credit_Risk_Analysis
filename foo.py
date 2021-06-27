#!/usr/bin/env python

import numpy as np


V = np.array(
    [0.01057991, 0.0296696 , 0.01980243, 0.01483256, 0.01747063,
    0.00308285, 0.00621641, 0.00738223, 0.00201526, 0.01492666,
    0.01020921, 0.01488107, 0.01685829, 0.05883807, 0.05625614,
    0.07876809, 0.05355513, 0.00699883, 0.        , 0.        ,
    0.05003318, 0.00071503, 0.        , 0.        , 0.00419858,
    0.0128559 , 0.00975384, 0.00748972, 0.00486002, 0.00656143,
    0.01353413, 0.01196899, 0.01311545, 0.00401892, 0.00563455,
    0.01489935, 0.01304461, 0.01336476, 0.0062406 , 0.00608322,
    0.01277358, 0.0067145 , 0.0133229 , 0.01326593, 0.01461382,
    0.00027308, 0.        , 0.01392109, 0.01485945, 0.00939335,
    0.01098295, 0.00617089, 0.01057536, 0.01522071, 0.00375719,
    0.00686666, 0.00963247, 0.00757054, 0.00872448, 0.00966205,
    0.00833097, 0.01032007, 0.00727267, 0.00862894, 0.        ,
    0.        , 0.00099569, 0.00754881, 0.01303005, 0.00513105,
    0.00137279, 0.        , 0.01448769, 0.01641297, 0.0130453 ,
    0.01279909, 0.0004956 , 0.00145897, 0.00282452, 0.00291332,
    0.00220495, 0.00150464, 0.00193258, 0.00564021, 0.02112913,
    0.01511524, 0.        , 0.00110057, 0.00115207, 0.005106  ,
    0.00503665, 0.0019016 , 0.00209124, 0.        , 0.        ])
F = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
    'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal',
    'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt',
    'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
    'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
    'last_pymnt_amnt', 'collections_12_mths_ex_med', 'policy_code',
    'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m',
    'open_act_il', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il',
    'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc',
    'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
    'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util',
    'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct',
    'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
    'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_inq',
    'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
    'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
    'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m',
    'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',
    'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec_bankruptcies',
    'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit',
    'total_il_high_credit_limit', 'home_ownership_ANY',
    'home_ownership_MORTGAGE', 'home_ownership_OWN', 'home_ownership_RENT',
    'verification_status_Not Verified',
    'verification_status_Source Verified', 'verification_status_Verified',
    'issue_d_Feb-2019', 'issue_d_Jan-2019', 'issue_d_Mar-2019',
    'pymnt_plan_n', 'initial_list_status_f', 'initial_list_status_w',
    'next_pymnt_d_Apr-2019', 'next_pymnt_d_May-2019',
    'application_type_Individual', 'application_type_Joint App',
    'hardship_flag_N', 'debt_settlement_flag_N']

z = zip(V, F)
#for elem in sorted(z, key=lambda x: x[1], reverse=False):
for elem in z:
    print(elem)
