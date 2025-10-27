import numpy as np
import torch
from utils.utils import *
from utils.loss import *
import os
from dataset_modules.dataset_generic import save_splits
from models.model import *
from sklearn.preprocessing import label_binarize
from sklearn.metrics import *
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=50, stop_epoch=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_min = np.Inf

    def __call__(self, epoch, score, model, ckpt_name = 'checkpoint.pt'):

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(score, model, ckpt_name)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print(f'epoch: {epoch} self.stop_epoch: {self.stop_epoch}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True

    def save_checkpoint(self, score, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'val_F1: ({self.score_min:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.score_min = score


def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit loss function...', end=' ')
    # if args.bag_loss == 'svm':
    #     from topk.svm import SmoothTop1SVM
    #     loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
    #     if device.type == 'cuda':
    #         loss_fn = loss_fn.cuda()
    if args.bag_loss == 'focal':
        loss_fn = FocalLoss2(alpha=0.25, gamma=2)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb', 'msamil', 'msdlm', 'abmil', 'dsmil', 'transmil']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            # from topk.svm import SmoothTop1SVM
            # instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            # if device.type == 'cuda':
            #     instance_loss_fn = instance_loss_fn.cuda()
            instance_loss_fn = None
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type == 'clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'msdlm':
            model = MSDLM(dropout=args.drop_out, n_classes=args.n_classes, n_modalities=args.n_modalities, 
                          embed_dim=args.embed_dim, method=args.method)
        elif args.model_type == 'msamil':
            model = MSAMIL(mil_method=args.mil_method, dropout=args.drop_out, n_classes=args.n_classes, n_modalities=args.n_modalities, 
                          embed_dim=args.embed_dim)
        elif args.model_type == 'abmil':
            model = DeepMIL(embed_dim=args.embed_dim, n_classes=args.n_classes, dropout=args.drop_out)
        elif args.model_type == 'dsmil':
            model = DSMIL(embed_dim=args.embed_dim, n_classes=args.n_classes, dropout=args.drop_out)
        elif args.model_type == 'transmil':
            model = TransMIL(embed_dim=args.embed_dim, n_classes=args.n_classes)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(dropout = args.drop_out, n_classes = args.n_classes, embed_dim=args.embed_dim)
        else:
            model = MIL_fc(dropout = args.drop_out, n_classes = args.n_classes, embed_dim=args.embed_dim)
    
    _ = model.to(device)
    print('Done!')

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    if args.lr_scheduler:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=10,eta_min=0.0000001)
    else:
        lr_scheduler = None
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, lr_scheduler,
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    results_dict, val_error, val_auc, acc_logger = summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/val_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.close()
    return results_dict, val_auc, 1-val_error 


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    Y_hats = []
    labels = []

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label.repeat(logits.shape[0]))
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        # if (batch_idx + 1) % 20 == 0:
        #     print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
        #         'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        optimizer.zero_grad()
        total_loss.backward()
        # step
        optimizer.step()

        Y_hats.extend(Y_hat.cpu().squeeze(1))
        labels.extend(label.cpu())

    running_accuracy = accuracy_score(labels, Y_hats)
    running_precision = precision_score(labels, Y_hats)
    running_recall = recall_score(labels, Y_hats)
    running_F1macro = f1_score(labels, Y_hats, average='macro')
    running_F1 = f1_score(labels, Y_hats)

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))

    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    print('train_accuracy: {:.4f}, train_F1macro:  {:.4f}, train_F1: {:.4f}'.format(running_accuracy, running_F1macro, running_F1))
    print('')


    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)
        writer.add_scalar('train/accuracy', running_accuracy, epoch)
        writer.add_scalar('train/precision', running_precision, epoch)
        writer.add_scalar('train/recall', running_recall, epoch)
        writer.add_scalar('train/F1macro', running_F1macro, epoch)
        writer.add_scalar('train/F1micro', running_F1, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    Y_hats = []
    labels = []

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label.repeat(logits.shape[0]))              # [n_patches, n_classes]
        # print("Y_prob.shape, label.shape: ", Y_prob.shape, label.shape)    # [n_classes], [1]
        # loss = loss_fn(Y_prob.unsqueeze(0), label)
        loss_value = loss.item()
        
        train_loss += loss_value
        # if (batch_idx + 1) % 20 == 0:
        #     print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

        if Y_hat.shape == torch.Size([1]):
            Y_hats.extend(Y_hat.cpu())
        else:
            Y_hats.extend(Y_hat.cpu().squeeze(1))
        # Y_hats.extend(Y_hat.cpu())
        labels.extend(label.cpu())


    running_accuracy = accuracy_score(labels, Y_hats)
    running_precision = precision_score(labels, Y_hats)
    running_recall = recall_score(labels, Y_hats)
    running_F1macro = f1_score(labels, Y_hats, average='macro')
    running_F1 = f1_score(labels, Y_hats)

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, lr: {:.8f}'.format(epoch, train_loss, train_error, optimizer.state_dict()['param_groups'][0]['lr']))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    print('train_accuracy: {:.4f}, train_F1macro:  {:.4f}, train_F1: {:.4f}'.format(running_accuracy, running_F1macro, running_F1))
    print('')

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/accuracy', running_accuracy, epoch)
        writer.add_scalar('train/precision', running_precision, epoch)
        writer.add_scalar('train/recall', running_recall, epoch)
        writer.add_scalar('train/F1macro', running_F1macro, epoch)
        writer.add_scalar('train/F1', running_F1, epoch)

   
def validate(cur, epoch, model, loader, n_classes, lr_scheduler, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.

    Y_hats = []
    true_labels = []
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            # loss = loss_fn(logits, label)
            loss = loss_fn(logits, label.repeat(logits.shape[0]))

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error

            if Y_hat.shape == torch.Size([1]):
                Y_hats.extend(Y_hat.cpu())
            else:
                Y_hats.extend(Y_hat.cpu().squeeze(1))
            # Y_hats.extend(Y_hat.cpu())
            true_labels.extend(label.cpu())
            

    val_accuracy = accuracy_score(true_labels, Y_hats)
    val_precision = precision_score(true_labels, Y_hats)
    val_recall = recall_score(true_labels, Y_hats)
    val_F1macro = f1_score(true_labels, Y_hats, average='macro')
    val_F1 = f1_score(true_labels, Y_hats)

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/accuracy', val_accuracy, epoch)
        writer.add_scalar('val/precision', val_precision, epoch)
        writer.add_scalar('val/recall', val_recall, epoch)
        writer.add_scalar('val/F1macro', val_F1macro, epoch)
        writer.add_scalar('val/F1', val_F1, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))  

    print('val_accuracy: {:.4f}, val_F1macro:  {:.4f}, val_F1: {:.4f}'.format(val_accuracy, val_F1macro, val_F1))
    print('')   

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_F1, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True
    if lr_scheduler is not None:
        lr_scheduler.step()

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0

    Y_hats = []
    true_labels = []
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label.repeat(logits.shape[0]))

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

            Y_hats.extend(Y_hat.cpu().squeeze(1))
            true_labels.extend(label.cpu())


    val_accuracy = accuracy_score(true_labels, Y_hats)
    val_precision = precision_score(true_labels, Y_hats)
    val_recall = recall_score(true_labels, Y_hats)
    val_F1macro = f1_score(true_labels, Y_hats, average='macro')
    val_F1 = f1_score(true_labels, Y_hats)

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))

    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)
        writer.add_scalar('val/accuracy', val_accuracy, epoch)
        writer.add_scalar('val/precision', val_precision, epoch)
        writer.add_scalar('val/recall', val_recall, epoch)
        writer.add_scalar('val/F1macro', val_F1macro, epoch)
        writer.add_scalar('val/F1', val_F1, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

    print('val_accuracy: {:.4f}, val_F1macro:  {:.4f}, val_F1: {:.4f}'.format(val_accuracy, val_F1macro, val_F1))
    print('')   
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_F1, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    val_loss = 0.
    val_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    patient_ids = loader.dataset.patient_data['Patient_ID']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        patient_id = patient_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({patient_id: {'Patient_ID': np.array(patient_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        val_error += error

    val_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, val_error, auc, acc_logger
