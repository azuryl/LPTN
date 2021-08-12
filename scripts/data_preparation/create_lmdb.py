from codes.utils import scandir
from codes.utils.lmdb_util import make_lmdb_from_imgs


def create_lmdb():
    '''
    folder_path = 'datasets/delight/delight_480p/train/A'  #'datasets/FiveK/FiveK_480p/train/A'
    lmdb_path = 'datasets/delight/delight_train_source.lmdb'#'datasets/FiveK/FiveK_train_source.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
    
    folder_path = 'datasets/delight/delight_480p/train/B'#'datasets/FiveK/FiveK_480p/train/B'
    lmdb_path = 'datasets/delight/delight_train_target.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
    '''
    '''
    folder_path = 'datasets/delight/delight_480p/test/A'#'datasets/FiveK/FiveK_480p/test/A'
    lmdb_path = 'datasets/delight/delight_test_source.lmdb'#'datasets/FiveK/FiveK_test_source.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = 'datasets/delight/delight_480p/test/B'#'datasets/FiveK/FiveK_480p/test/B'
    lmdb_path = 'datasets/delight/delight_test_target.lmdb'#'datasets/FiveK/FiveK_test_target.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
    '''
    '''
    folder_path = 'datasets/diva/diva_480p/train/A'#229'datasets/FiveK/FiveK_480p/test/A'
    lmdb_path = 'datasets/diva/diva_train_source.lmdb'#'datasets/FiveK/FiveK_test_source.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = 'datasets/diva/diva_480p/train/B'#330datasets/FiveK/FiveK_480p/test/B'
    lmdb_path = 'datasets/diva/diva_train_target.lmdb'#'datasets/FiveK/FiveK_test_target.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
    '''
    '''
    folder_path = 'datasets/diva/diva_480p/test/A'#'datasets/FiveK/FiveK_480p/test/A'
    lmdb_path = 'datasets/diva/diva_test_source.lmdb'#'datasets/FiveK/FiveK_test_source.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = 'datasets/diva/diva_480p/test/B'#'datasets/FiveK/FiveK_480p/test/B'
    lmdb_path = 'datasets/diva/diva_test_target.lmdb'#'datasets/FiveK/FiveK_test_target.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
    '''
    '''
    folder_path = 'datasets/diva/diva_480p/test/G299test'#'datasets/FiveK/FiveK_480p/test/A'
    lmdb_path = 'datasets/diva/diva_G299test_source.lmdb'#'datasets/FiveK/FiveK_test_source.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = 'datasets/diva/diva_480p/test/G330test'#'datasets/FiveK/FiveK_480p/test/B'
    lmdb_path = 'datasets/diva/diva_G330test_target.lmdb'#'datasets/FiveK/FiveK_test_target.lmdb'
    '''
    folder_path = 'datasets/school/school_480p/train/299p'#330datasets/FiveK/FiveK_480p/test/B'
    lmdb_path = 'datasets/school/school_train_source.lmdb'#'datasets/FiveK/FiveK_test_target.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = 'datasets/school/school_480p/train/330p'#330datasets/FiveK/FiveK_480p/test/B'
    lmdb_path = 'datasets/school/school_train_target.lmdb'#'datasets/FiveK/FiveK_test_target.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def prepare_keys(folder_path):

    print('Reading image path list ...',folder_path)
    img_path_list = sorted(
        list(scandir(folder_path, suffix='png', recursive=False)))
    print("img_path_list :",img_path_list)
    import sys
    #sys.exit()
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]
    for img_path in sorted(img_path_list):
        key = img_path.split('.png')[0]
        #print("!!!!!!!!!!create_lamdb.py key:",key)
       
    return img_path_list, keys


if __name__ == '__main__':

    create_lmdb()

