from astropy.io import fits
import numpy as np
import os
import pandas as pd
from label_maps import fine_label_map, coarse_label_map
from utils import normalise_features


def create_dataset(args):
    print("Building dataset")

    file_sets = [
        {"name": 'small_magellanic_cloud', 'label_type': 'fine', 'kind': 'train', 'active': True},
        {"name": 'large_magellanic_cloud', 'label_type': 'fine', 'kind': 'train', 'active': True},
        {"name": 'galactic_plane', 'label_type': 'coarse',
         'kind': 'test' if args.test_set == 'galactic_plane' else 'train', 'active': True},
        {"name": 'bulge', 'label_type': 'none', 'kind': 'test', 'active': True if args.test_set == 'bulge' else 'False'},
    ]

    expected_flux_data_length = 387

    train_features = []
    train_labels = []
    train_aors = []
    train_indices = []
    test_features = []
    test_labels = []
    test_aors = []
    test_indices = []
    for file_set in file_sets:
        if file_set['active'] != True:
            continue
        aors = np.genfromtxt(os.path.join(args.aors_dir, file_set["name"] + '.csv'),
                             delimiter=',', skip_header=1, usecols=(0), dtype=str)
        indices = np.genfromtxt(os.path.join(args.aors_dir, file_set["name"] + '.csv'),
                                delimiter=',', skip_header=1, usecols=(1), dtype=str)
        text_labels = np.genfromtxt(os.path.join(args.aors_dir, file_set["name"] + '.csv'),
                                    delimiter=',', skip_header=1, usecols=(2), dtype=str)
        for aor, index, label in zip(aors, indices, text_labels):
            if index == 'None':
                file_path = os.path.join(args.fits_dir, file_set["name"], 'cassis_yaaar_spcfw_{}t.fits'.format(aor))
                if not os.path.exists(file_path):
                    file_path = os.path.join(args.fits_dir, file_set["name"], 'cassis_yaaar_sptfc_{}t.fits'.format(aor))
                    if not os.path.exists(file_path):
                        print("Fits file does not exist for: AOR={}, Pointing={} from file set {}.".format(aor, index, file_set['name']))
                        continue
            else:
                file_path = os.path.join(args.fits_dir, file_set["name"], 'cassis_yaaar_spcfw_{}_{}t.fits'.format(aor, int(index)))
                if not os.path.exists(file_path):
                    file_path = os.path.join(
                        args.fits_dir, file_set["name"],
                        'cassis_yaaar_sptfc_{}_{}t.fits'.format(aor, int(index)))
                    if not os.path.exists(file_path):
                        print("Fits file does not exist for: AOR={}, Pointing={} from file set {}.".format(aor,index, file_set['name']))
                        continue

            with fits.open(fr'{file_path}') as hdu:
                wavelength_data = np.array(hdu[0].data[:,0])
                flux_data = np.array(hdu[0].data[:,1])

                # check for invalid data
                if np.any(np.isnan(flux_data)):
                    if args.verbose == 'on':
                        print("Flux data contains NaNs for: AOR={}, Pointing={}. Skipping file.".format(aor, index))
                    continue

                # normalize the data
                flux_data = normalise_features(flux_data)

                # ideally the spectrum contains 387 elements, if it doesn't attempt to pad the data
                features_length = len(wavelength_data)
                if features_length != expected_flux_data_length:  # repair the data
                    if args.verbose == 'on':
                        print(file_set['name'], aor, index, label,  len(wavelength_data), min(wavelength_data),
                              max(wavelength_data), flux_data[0], flux_data[-1])
                    continue
                if file_set['kind'] == 'train':
                    train_features.append(flux_data)
                    if file_set["label_type"] == 'fine':
                        train_labels.append(fine_label_map[label]['coarse_index'])
                    elif file_set["label_type"] == 'coarse':
                        train_labels.append(coarse_label_map.index(label))
                    else:
                        raise ValueError('Training data should have a label kind "fine" or "coarse".')
                    train_aors.append(aor)
                    train_indices.append(index)
                else:  # test
                    test_features.append(flux_data)
                    if file_set["label_type"] == 'fine':
                        test_labels.append(fine_label_map[label]['coarse_index'])
                    elif file_set["label_type"] == 'coarse':
                        test_labels.append(coarse_label_map.index(label))
                    else:
                        test_labels.append(-1)
                    test_aors.append(aor)
                    test_indices.append(index)

    os.makedirs(args.training_dir, exist_ok=True)
    df_train_features = pd.DataFrame(np.array(train_features))
    df_train_features.to_csv(os.path.join(args.training_dir, "train_features.csv"), header=False, index=False)
    df_train_labels = pd.DataFrame({'aor': np.array(train_aors), 'index': np.array(train_indices),
                                    'label': np.array(train_labels)})
    df_train_labels.to_csv(os.path.join(args.training_dir, "train_labels.csv"), header=True, index=False)

    if len(test_features) > 0:
        os.makedirs(args.test_dir, exist_ok=True)
        df_test_features = pd.DataFrame(np.array(test_features))
        df_test_features.to_csv(os.path.join(args.test_dir, "test_features.csv"), header=False, index=False)
        df_test_labels = pd.DataFrame({'aor': np.array(test_aors), 'index': np.array(test_indices),
                                       'label': np.array(test_labels)})
        df_test_labels.to_csv(os.path.join(args.test_dir, "test_labels.csv"), header=True, index=False)
