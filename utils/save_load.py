import os
import torch as t

def model_save(model, save_path, epoch_label):
    '''
    Save Model "Name+Epoch"
    '''
    save_filename = (model.model_name + '_epo%s.pth' % epoch_label)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    t.save(model.state_dict(),os.path.join(save_path,save_filename))
    print('Model:'+ save_filename+ ' saves successfully' )

def model_load(model, load_path, epoch_label, dont_load_clf_weights=False, verbose=True):
    '''
    Load Model
    '''
    save_filename = (model.model_name + '_epo%s.pth' % epoch_label)
    loaded_state_dict = t.load(os.path.join(load_path,save_filename),map_location=t.device('cpu'))
    if dont_load_clf_weights:
        print('Excluding classifier weights in loading of model...')
        loaded_state_dict = {k:v for k,v in loaded_state_dict.items() if 'classifier' not in k}
    model_dict = model.state_dict()
    model_dict.update(loaded_state_dict)
    model.load_state_dict(model_dict)
    # model.load_state_dict(t.load(os.path.join(load_path,save_filename),map_location=t.device('cpu')))
    if verbose:
        print('Model:'+ save_filename+ ' loads successfully' )

