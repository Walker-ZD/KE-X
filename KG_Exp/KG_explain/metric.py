
def calculate_metric(node_dict):

    GE_fidelity_Acc_list = []
    GE_fidelity_Prob_list = []
    GE_sparity_list = []

    SM_fidelity_Acc_list = []
    SM_fidelity_Prob_list = []
    SM_sparity_list = []

    GC_fidelity_Acc_list = []
    GC_fidelity_Prob_list = []
    GC_sparity_list = []

    for key in node_dict:

        node = node_dict[key]
        GE_fidelity_Acc_list.append(node['same_GE'])
        GE_fidelity_Prob_list.append(node['pred_decay_GE'])
        GE_sparity_list.append(node['sparity_GE'])

        SM_fidelity_Acc_list.append(node['same_SM'])
        SM_fidelity_Prob_list.append(node['pred_decay_SM'])
        SM_sparity_list.append(node['sparity_SM'])

        GC_fidelity_Acc_list.append(node['same_GC'])
        GC_fidelity_Prob_list.append(node['pred_decay_GC'])
        GC_sparity_list.append(node['sparity_GC'])


    GE_fidelity_Prob = sum(GE_fidelity_Prob_list) / len(GE_fidelity_Prob_list)
    GE_fidelity_Acc = 1 - sum(GE_fidelity_Acc_list) / len(GE_fidelity_Acc_list)
    GE_sparity = sum(GE_sparity_list) / len(GE_sparity_list)

    SM_fidelity_Prob = sum(SM_fidelity_Prob_list) / len(SM_fidelity_Prob_list)
    SM_fidelity_Acc = 1 - sum(SM_fidelity_Acc_list) / len(SM_fidelity_Acc_list)
    SM_sparity = sum(SM_sparity_list) / len(SM_sparity_list)

    GC_fidelity_Prob = sum(GC_fidelity_Prob_list) / len(GC_fidelity_Prob_list)
    GC_fidelity_Acc = 1 - sum(GC_fidelity_Acc_list) / len(GC_fidelity_Acc_list)
    GC_sparity = sum(GC_sparity_list) / len(GC_sparity_list)

    print("GE_fidelity_Prob: " + str(GE_fidelity_Prob) + "\n"
        + "GE_fidelity_Acc: " + str(GE_fidelity_Acc) + "\n"
        + "SM_fidelity_Prob: " + str(SM_fidelity_Prob) + "\n"
        + "SM_fidelity_Acc: " + str(SM_fidelity_Acc) + "\n"
        + "GC_fidelity_Prob: " + str(GC_fidelity_Prob) + "\n"
        + "GC_fidelity_Acc: " + str(GC_fidelity_Acc) + "\n\n"
        + "GE_sparity: " + str(GE_sparity) + "\n"
        + "SM_sparity: " + str(SM_sparity) + "\n"
        + "GC_sparity: " + str(GC_sparity) + "\n")
    
    result_dict = {
        "GE_fidelity_Prob": GE_fidelity_Prob,
        "GE_fidelity_Acc": GE_fidelity_Acc,
        "SM_fidelity_Prob": SM_fidelity_Prob,
        "SM_fidelity_Acc": SM_fidelity_Acc,
        "GC_fidelity_Prob": GC_fidelity_Prob,
        "GC_fidelity_Acc": GC_fidelity_Acc,
        "GE_sparsity": GE_sparity,
        "SM_sparsity": SM_sparity,
        "GC_sparsity": GC_sparity
    }

    return result_dict