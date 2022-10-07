import torch
from torch.utils import data
from GCN.config import shadow_dir, mask_dir, pth
from GCN._1_mcdo import MCDO
from GCN._2_generate_ROI import generate_ROI
from GCN._3_Creating_graph import creating_graph
from GCN._4_GCN_training import GCN_training
from GCN._5_GCN_inference import GCN_inference
from model.ResNet import ResNet as MyModel
from dataset import Dataset_train as Dataset

if __name__ == '__main__':
    # 测试数据路径
    testing_dataset = Dataset(shadow_dir, mask_dir)
    testing_dl = data.DataLoader(testing_dataset, batch_size=1)
    number_of_samples = len(shadow_dir)
    for i, (shadow, mask) in enumerate(testing_dl):
        print("-------------%3s/%s image processing-------------" % (i+1, number_of_samples))
        # step 0 加载模型
        model = MyModel().cuda()
        model.load_state_dict(torch.load(pth))
        # step1 MCDO
        ref_shape, shadow, GT = MCDO(model, 20, shadow, mask)
        # step2 生成ROI
        generate_ROI(ref_shape)
        # step3 构建Graphs
        creating_graph()
        # step4 GCN training
        model, adj, features, y_test, idx_test = GCN_training(epochs=3000, dropout=0.3)
        # step5 GCN inference
        GCN_inference(shadow, GT, model, adj, features, get_probs=False, gcn_th=0.3, i=i)
