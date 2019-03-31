import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
sns.set_style("darkgrid")

import cv2
import xgboost as xgb

from ml_metrics import quadratic_weighted_kappa

import difflib

import matplotlib.pyplot as plt
from pandas.io.json import json_normalize

import os
import warnings

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
import json
import scipy as sp
import langid
from PIL import Image
from functools import partial
from math import sqrt
from collections import Counter
import lightgbm as lgb

np.random.seed(369)
from os.path import join as jp

# Any results you write to the current directory are saved as output.
input_dir = jp(os.pardir, 'input')

warnings.simplefilter(action='ignore', category=FutureWarning)
train_data_path = "../input/petfinder-adoption-prediction/train/train.csv"
test_data_path = '../input/petfinder-adoption-prediction/test/test.csv'


# Global function
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']


def impact_coding(data, feature, target='y'):
    from sklearn.model_selection import KFold

    '''
    In this implementation we get the values and the dictionary as two different steps.
    This is just because initially we were ignoring the dictionary as a result variable.

    In this implementation the KFolds use shuffling. If you want reproducibility the cv
    could be moved to a parameter.
    '''
    n_folds = 20
    n_inner_folds = 10
    impact_coded = pd.Series()

    oof_default_mean = data[target].mean()  # Gobal mean to use by default (you could further tune this)
    kf = KFold(n_splits=n_folds, shuffle=True)
    oof_mean_cv = pd.DataFrame()
    split = 0
    for infold, oof in kf.split(data[feature]):
        impact_coded_cv = pd.Series()
        kf_inner = KFold(n_splits=n_inner_folds, shuffle=True)
        inner_split = 0
        inner_oof_mean_cv = pd.DataFrame()
        oof_default_inner_mean = data.iloc[infold][target].mean()
        for infold_inner, oof_inner in kf_inner.split(data.iloc[infold]):
            # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)
            oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()
            impact_coded_cv = impact_coded_cv.append(data.iloc[infold].apply(
                lambda x: oof_mean[x[feature]]
                if x[feature] in oof_mean.index
                else oof_default_inner_mean
                , axis=1))

            # Also populate mapping (this has all group -> mean for all inner CV folds)
            inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
            inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)
            inner_split += 1

        # Also populate mapping
        oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
        oof_mean_cv.fillna(value=oof_default_mean, inplace=True)
        split += 1

        impact_coded = impact_coded.append(data.iloc[oof].apply(
            lambda x: inner_oof_mean_cv.loc[x[feature]].mean()
            if x[feature] in inner_oof_mean_cv.index
            else oof_default_mean
            , axis=1))

    return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean

def frequency_encoding(df, col_name):
    new_name = "{}_counts".format(col_name)
    new_col_name = "{}_freq".format(col_name)
    grouped = df.groupby(col_name).size().reset_index(name=new_name)
    df = df.merge(grouped, how="left", on=col_name)
    df[new_col_name] = df[new_name] / df[new_name].count()
    del df[new_name]
    return df


def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


def detect_language(description):
    lang = langid.classify(description)[0]
    if lang in ['fi', 'lb', 'pl', 'nb', 'eo', 'et', 'pt', 'lt',
                'no', 'de', 'tl', 'nl', 'da', 'ro', 'fr', 'it',
                'hr', 'la', 'sw', 'es', 'mg', 'mt', 'sl', 'eu',
                'sv', 'ca', 'cs', 'sk', 'xh', 'hu']:
        lang = 'en'
    if lang in ['af', 'ms', 'jv', 'ja', 'bs']:
        lang = 'id'
    return lang


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# state GDP: https://en.wikipedia.org/wiki/List_of_Malaysian_states_by_GDP
state_gdp = {
    41336: 116.679,
    41325: 40.596,
    41367: 23.02,
    41401: 190.075,
    41415: 5.984,
    41324: 37.274,
    41332: 42.389,
    41335: 52.452,
    41330: 67.629,
    41380: 5.642,
    41327: 81.284,
    41345: 80.167,
    41342: 121.414,
    41326: 280.698,
    41361: 32.270
}

# state population: https://en.wikipedia.org/wiki/Malaysia
state_population = {
    41336: 33.48283,
    41325: 19.47651,
    41367: 15.39601,
    41401: 16.74621,
    41415: 0.86908,
    41324: 8.21110,
    41332: 10.21064,
    41335: 15.00817,
    41330: 23.52743,
    41380: 2.31541,
    41327: 15.61383,
    41345: 32.06742,
    41342: 24.71140,
    41326: 54.62141,
    41361: 10.35977
}

color_label = {
    0: 'empty',
    1: 'Black',
    2: 'Brown',
    3: 'Golden',
    4: 'Yellow',
    5: 'Cream',
    6: 'Gray',
    7: 'White'
}
state_label = {
41336:'Johor',
41325:'Kedah',
41367:'Kelantan',
41401:'Kuala Lumpur',
41415:'Labuan',
41324:'Melaka',
41332:'Negeri Sembilan',
41335:'Pahang',
41330:'Perak',
41380:'Perlis',
41327:'Pulau Pinang',
41345:'Sabah',
41342:'Sarawak',
41326:'Selangor',
41361:'Terengganu',
}
contact_mean = ['whatsapp', 'dm', 'pm', 'text', 'messenger', 'email', 'sms', 'phone', 'call']


def get_contacts_source(description):
    return


print('[INFO] Import Done')


def extract_features(df):
    print('here______________________________')
    from tensorflow.keras.applications.densenet import preprocess_input, DenseNet121
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
    import tensorflow.keras.backend as K
    import tensorflow as tf
    from tensorflow.python.client import device_lib

    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.Session()

    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    print(get_available_gpus())

    def get_image_path(adoptionspeed):
        if np.isnan(adoptionspeed):
            return 'petfinder-adoption-prediction/test_images'
        else:
            return 'petfinder-adoption-prediction/train_images'

    def load_image(path, img_size):
        def resize_to_square(im, img_size):
            old_size = im.shape[:2]  # old_size is in (height, width) format
            ratio = float(img_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            # new_size should be in (width, height) format
            im = cv2.resize(im, (new_size[1], new_size[0]))
            delta_w = img_size - new_size[1]
            delta_h = img_size - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            color = [0, 0, 0]
            new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            return new_im

        image = cv2.imread(path)
        new_image = resize_to_square(image, img_size)
        new_image = preprocess_input(new_image)
        return new_image

    def init_densenet():
        print('[INFO] Init Densenet...')
        inp = Input((256, 256, 3))
        print('[INFO] import Densenet')
        backbone = DenseNet121(input_tensor=inp, include_top=False,
                               weights='../input/densenet-121-weights/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
        print('[INFO] import Densenet DONE')
        x = backbone.output
        x = GlobalAveragePooling2D()(x)
        x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
        x = AveragePooling1D(4)(x)
        out = Lambda(lambda x: x[:, :, 0])(x)
        m = Model(inp, out)
        print('[INFO] Init Densenet DONE.')
        return m

    m = init_densenet()

    print('[INFO] Start Image Features_Extraction...')
    extract_features_path = jp(input_dir, 'image_extraction_features.csv')
    if os.path.isfile(extract_features_path):
        print('[INFO] File in the system find OK')
        df_features = pd.read_csv(extract_features_path, sep=";")
        if df_features.shape[0] == df.shape[0] and df_features['PetID'].equals(df['PetID']):
            print('[INFO] File in the system with the same size and PetID OK')
            df = pd.concat([df.set_index('PetID'), df_features.set_index('PetID')], sort=False, axis=1).reset_index()
            df.rename(columns={df.columns[0]: "PetID"}, inplace=True)
            print('[INFO] Image Features_Extraction DONE.')
            return df
    print('[INFO] File saved does not match')
    img_size = 256
    batch_size = 16
    pet_ids = df['PetID'].values
    adoptionspeed_list = df['AdoptionSpeed'].values
    n_batches = len(pet_ids) // batch_size + 1
    features = {}
    for b in tqdm(range(n_batches)):
        start = b * batch_size
        end = (b + 1) * batch_size
        batch_pets = pet_ids[start:end]
        batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
        for i, (pet_id, adoptionspeed_) in enumerate(zip(batch_pets, adoptionspeed_list)):

            subfolder = get_image_path(adoptionspeed_)
            # _photos = int(df[df['PetID']==pet_id].PhotoAmt)
            # print(_photos)
            image_name = '{}-{}.jpg'.format(pet_id, 1)
            image_path = jp(input_dir, subfolder, image_name)
            if os.path.isfile(image_path):
                batch_images[i] = load_image(image_path, 256)
        batch_preds = m.predict(batch_images)
        for i, pet_id in enumerate(batch_pets):
            features[pet_id] = batch_preds[i]

    df_features = pd.DataFrame.from_dict(features, orient='index')
    df_features.rename(columns=lambda k: 'img_{}'.format(k), inplace=True)
    df_features.reset_index(inplace=True)
    df_features.rename(columns={df_features.columns[0]: "PetID"}, inplace=True)
    n_components = 200
    svd = TruncatedSVD(n_components=n_components)
    X = df_features[['img_{}'.format(k) for k in range(256)]].values
    svd.fit(X)
    print('fit done')
    X_svd = svd.transform(X)
    X_svd = pd.DataFrame(X_svd, columns=['img_svd_{}'.format(i) for i in range(n_components)])
    X_svd['PetID'] = df.PetID.values.tolist()



    df = pd.concat([df.set_index('PetID'), X_svd.set_index('PetID')], sort=False, axis=1).reset_index()
    df.rename(columns={df.columns[0]: "PetID"}, inplace=True)
    print('[INFO] Image Features_Extraction DONE.')
    return df


def features_engineering(dfs):
    def get_sentiments(df, subfolder):
        sentiments = []
        pet_ids = df.PetID
        with tqdm(total=len(pet_ids), desc='Reading sentiment') as pbar:
            for pet_id in pet_ids:
                result = {'PetID': pet_id, 'sent_magnitude': 0.0, 'sent_score': 0.0, 'sent_number_of_sentence': 0,
                          'sent_number_of_entities': 0, 'entities': ' '}
                filepath = jp(input_dir, subfolder, '{}.json'.format(pet_id))
                if os.path.isfile(filepath):
                    with open(filepath, encoding="utf8") as f:
                        data = json.load(f)
                        result['entities'] = result['entities'].join([x['name'] for x in data['entities']])
                        result['sent_magnitude'] = data['documentSentiment']['magnitude']
                        result['sent_score'] = data['documentSentiment']['score']
                        result['sent_number_of_sentence'] = len(data['sentences'])
                        result['sent_number_of_entities'] = len(data['entities'])
                sentiments.append(result)
                pbar.update()
        df = pd.concat([df.set_index('PetID'), pd.DataFrame.from_dict(sentiments).set_index('PetID')],
                       sort=False, axis=1).reset_index()
        df.rename(columns={df.columns[0]: "PetID"}, inplace=True)
        return df

    def get_image_details(df, subfolder):
        from joblib import Parallel, delayed

        def myfun(idnex, row):
            _id = row.PetID
            if np.isnan(row.PhotoAmt):
                _photos = 0
            else:
                _photos = int(row.PhotoAmt)
            count = 0
            result = {'PetID': _id, 'width': 0, 'height': 0, 'resolution': 0, 'vector_mean': 0, 'vector_std': 0}
            for i in range(_photos):
                image_name = '{}-{}.jpg'.format(_id, str(i + 1))
                image_path = jp(input_dir, subfolder, image_name)
                if os.path.isfile(image_path):
                    image = Image.open(image_path)
                    size = image.size
                    vector = np.asarray(image)
                    result['vector_mean'] += np.mean(vector)
                    result['vector_std'] += np.std(vector)
                    result['width'] += size[0]
                    result['height'] += size[1]
                    count += 1
            if count > 0:
                result['width'] = result['width'] / _photos
                result['height'] = result['height'] / _photos
                result['vector_mean'] = result['vector_mean'] / _photos
                result['vector_std'] = result['vector_std'] / _photos
                result['resolution'] = result['width'] * result['height']
            return result

        with tqdm(total=len(df), desc='Reading Images') as pbar:
            image_details = Parallel(n_jobs=-1, verbose=-1, backend="threading")(
                delayed(myfun)(index, row) for index, row in df.iterrows())
            pbar.update()

        df = pd.concat([df.set_index('PetID'), pd.DataFrame(image_details).set_index('PetID')],
                       sort=False, axis=1).reset_index()
        df.rename(columns={df.columns[0]: "PetID"}, inplace=True)

        return df

    def get_metadata(df, subfolder):
        # color details extraction
        vertex_xs = []
        vertex_ys = []
        bounding_confidences = []
        bounding_importance_fracs = []
        dominant_blues = []
        dominant_greens = []
        dominant_reds = []
        dominant_pixel_fracs = []
        dominant_scores = []
        label_descriptions = []
        label_scores = []
        label_topicalities = []
        nf_count = 0
        nl_count = 0

        confidence_mean = []
        pixel_fraction_mean = []
        score_mean = []
        importance_mean = []

        print('[INFO] Getting metadata')
        for pet in tqdm(df.PetID):
            try:
                # Metadata
                metadata_path = jp(input_dir, subfolder, str(pet) + '-1.json')
                with open(metadata_path, 'r', errors='ignore') as f:
                    data = json.load(f)
                vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
                vertex_xs.append(vertex_x)
                vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
                vertex_ys.append(vertex_y)

                bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
                bounding_confidences.append(bounding_confidence)

                dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
                dominant_pixel_fracs.append(dominant_pixel_frac)

                dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
                dominant_scores.append(dominant_score)

                file_colors = data['imagePropertiesAnnotation']['dominantColors']['colors']
                file_crops = data['cropHintsAnnotation']['cropHints']

                confidence_mean.append(np.asarray([x['score'] for x in file_colors]).mean())
                pixel_fraction_mean.append(np.asarray([x['pixelFraction'] for x in file_colors]).mean())
                score_mean.append(np.asarray([x['confidence'] for x in file_crops]).mean())
                if 'importanceFraction' in file_crops[0].keys():
                    importance_mean.append(np.asarray([x['importanceFraction'] for x in file_crops]).mean())
                else:
                    importance_mean.append(-1)

                bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
                bounding_importance_fracs.append(bounding_importance_frac)

                dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
                dominant_blues.append(dominant_blue)
                dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
                dominant_greens.append(dominant_green)
                dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
                dominant_reds.append(dominant_red)

                if data.get('labelAnnotations'):
                    label_description_all_tmp = ' '.join([x['description'] for x in data['labelAnnotations']])

                    label_description = data['labelAnnotations'][0]['description']
                    label_descriptions.append(label_description)

                    label_score = data['labelAnnotations'][0]['score']
                    label_scores.append(label_score)

                    label_topicality = data['labelAnnotations'][0]['topicality']
                    label_topicalities.append(label_topicality)
                else:
                    nl_count += 1
                    label_scores.append(-1)
                    label_topicalities.append(-1)
                    label_descriptions.append('nothing')

            except FileNotFoundError:
                confidence_mean.append(-1)
                pixel_fraction_mean.append(-1)
                score_mean.append(-1)
                importance_mean.append(-1)

                nf_count += 1
                vertex_xs.append(-1)
                vertex_ys.append(-1)
                bounding_confidences.append(-1)
                bounding_importance_fracs.append(-1)
                dominant_blues.append(-1)
                dominant_greens.append(-1)
                dominant_reds.append(-1)
                dominant_pixel_fracs.append(-1)
                dominant_scores.append(-1)
                label_descriptions.append('nothing')
                label_scores.append(-1)
                label_topicalities.append(-1)

        df.loc[:, 'vertex_x'] = vertex_xs
        df.loc[:, 'vertex_y'] = vertex_ys
        df.loc[:, 'bounding_confidence'] = bounding_confidences
        df.loc[:, 'bounding_importance'] = bounding_importance_fracs
        df.loc[:, 'dominant_blue'] = dominant_blues
        df.loc[:, 'dominant_green'] = dominant_greens
        df.loc[:, 'dominant_red'] = dominant_reds
        df.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
        df.loc[:, 'dominant_score'] = dominant_scores
        df.loc[:, 'label_description'] = label_descriptions
        df.loc[:, 'label_score'] = label_scores
        df.loc[:, 'topicality'] = label_topicalities
        df.loc[:, 'confidence_mean'] = confidence_mean
        df.loc[:, 'pixel_fraction_mean'] = pixel_fraction_mean
        df.loc[:, 'score_mean'] = score_mean
        df.loc[:, 'importance_mean'] = importance_mean
        return df

    def get_rarity_of_a_breed_by_location(df):
        print('[INFO] Get Rarity')
        a = df.groupby(['Breed1', 'State'])['PetID'].count().reset_index()
        a.columns = ['Breed1', 'State', 'Breed1_name_COUNT_by_State']

        b = df.groupby(['State'])['PetID'].count().reset_index()
        b.columns = ['State', 'Pet_COUNT']

        c = a.merge(b, how="left", left_on="State", right_on='State')
        c['percentage_of_breed_in_the_state'] = c['Breed1_name_COUNT_by_State'] / c['Pet_COUNT'] * 100
        # c.drop('State',axis=1,inplace=True)
        df = df.merge(c, how='left', left_on=['Breed1', 'State'], right_on=['Breed1', 'State'])
        print('[INFO] Get Rarity DONE')
        return df

    breed_df = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
    mapping_breed = dict(breed_df[['BreedID', 'BreedName']].values)
    mapping_breed[0] = 'nothing'

    for i, (df_, kind) in enumerate(zip(dfs, ['train', 'test'])):
        df_['Breed1'] = df_.Breed1.fillna(0)
        df_['Breed2'] = df_.Breed2.fillna(0)
        df_['Breed1'] = np.where((df_['Breed1'] == 0) & (df_['Breed2'] != 0), df_['Breed2'], df_['Breed1'])
        df_['Breed2'] = np.where((df_['Breed2'] == df_['Breed1']), 0, df_['Breed2'])
        df_['Breed1_name'] = df_.Breed1.map(mapping_breed)
        df_['Breed2_name'] = df_.Breed2.map(mapping_breed)

        df_.PhotoAmt.fillna(0, inplace=True)
        df_ = get_metadata(df_, 'petfinder-adoption-prediction/' + kind + '_metadata')
        df_ = get_image_details(df_, 'petfinder-adoption-prediction/' + kind + '_images')
        df_ = get_sentiments(df_, 'petfinder-adoption-prediction/' + kind + '_sentiment')
        dfs[i] = df_
    df = dfs[0].append(dfs[1])
    del dfs,df_
    df['PureBreed'] = np.where((df['Breed2'] == 0), 1, 0)

    breed_df = pd.read_csv('../input/breed-details-augmented/breed_labels.csv',sep=';',header=0)
    mapping_breed = dict(breed_df[['BreedName','BreedID']].values)
    df['Breed1_augmented'] = df.Breed1_name.map(mapping_breed)
    df['Breed2_augmented'] = df.Breed2_name.map(mapping_breed)

    df['Color1'] = df.Color1.map(color_label)
    df['Color2'] = df.Color2.map(color_label)
    df['Color3'] = df.Color3.map(color_label)
    df = get_rarity_of_a_breed_by_location(df)

    df['In_Kuala_Lumpur'] = np.where(df['State'] == 41326, 1, 0)
    df["state_gdp"] = df.State.map(state_gdp)
    df["state_population"] = df.State.map(state_population)
    df['State'] = df.State.map(state_label)

    df['Breed2_augmented'].fillna(0)
    df['Breed1_augmented'].fillna(0)

    # Add breed details
    with open('../input/breed-details/rating.json', encoding="utf8") as f:
        data = json.load(f)
    def summarize_data(df):
        df = df.copy()
        df_cat = df[df['Adaptability'].isna()]
        df_cat.fillna(0,inplace=True)
        df_cat['Score'] =(df_cat["Affectionate with Family"]+df_cat["Amount of Shedding"]+
                               df_cat["Easy to Groom"]+ df_cat["General Health"]+df_cat["Intelligence"]+
                               df_cat["Kid Friendly"]+df_cat["Pet Friendly"]+df_cat["Potential for Playfulness"])/8.0

        df_dog= df[~df['Adaptability'].isna()]
        df_dog.fillna(0,inplace=True)
        df_dog['Score'] = (df_dog["Adaptability"]+df_dog["All Around Friendliness"]+
        df_dog["Exercise Needs"]+df_dog["Health Grooming"]+df_dog["Trainability"]+
                               df_dog["Adapts Well to Apartment Living"]+df_dog["Affectionate with Family"]+
        df_dog["Amount Of Shedding"]+df_dog["Dog Friendly"]+df_dog["Drooling Potential"]+
        df_dog["Easy To Groom"]+df_dog["Easy To Train"]+df_dog["Energy Level"]+
        df_dog["Friendly Toward Strangers"]+df_dog["General Health"]+df_dog["Good For Novice Owners"]+
        df_dog["Incredibly Kid Friendly Dogs"]+df_dog["Intelligence"]+df_dog["Intensity"]+
        df_dog["Potential For Mouthiness"]+df_dog["Potential For Playfulness"]+df_dog["Potential For Weight Gain"]+
        df_dog["Prey Drive"]+df_dog["Sensitivity Level"]+df_dog["Tendency To Bark Or Howl"]+df_dog["Tolerates Being Alone"]+\
                     df_dog["Tolerates Cold Weather"]+df_dog["Tolerates Hot Weather"])/27.0

        df["Size"].fillna(3,inplace=True)
        df_cat = df_cat.append(df_dog)
        df['Score'] = df_cat.Score
        df = df[['Score','Size']]
        return df

    keys = list(data.keys())
    df_breed_detail = pd.DataFrame()
    for i, key_ in enumerate(keys):
        df_tmp_key = json_normalize(data[key_])
        df_tmp_key['Breed_Name'] = key_
        df_tmp_key = df_tmp_key.set_index('Breed_Name')
        if i == 0:
            df_breed_detail = df_tmp_key
        else:
            df_breed_detail = df_breed_detail.append(df_tmp_key)
    del df_tmp_key

    df_breed_detail = summarize_data(df_breed_detail)
    map_breed = {}
    not_defined = list(set(df.Breed2_name.values.tolist() + df.Breed1_name.values.tolist()) - set(
        df_breed_detail.reset_index().Breed_Name.values.tolist()))
    for name_not_defined in not_defined:
        match = difflib.get_close_matches(name_not_defined, df_breed_detail.reset_index().Breed_Name.values.tolist())
        if len(match) > 0:
            match = match[0]
            map_breed[name_not_defined] = match
        else:
            type_ = df[df['Breed1_name'] == name_not_defined].Type.values.tolist()
            if len(type_) == 0:
                print('no match for', name_not_defined)
                continue
            type_ = type_[0]
            if type_ == 2:
                map_breed[name_not_defined] = 'Cat'
            else:
                map_breed[name_not_defined] = 'Dog'

    df_breed_detail.fillna(0, inplace=True)
    df_breed_detail.reset_index(inplace=True)


    df['Breed1_name_modified'] = df.Breed1_name.replace(map_breed)
    df['Breed2_name_modified'] = df.Breed2_name.replace(map_breed)

    df = pd.merge(df, df_breed_detail.add_prefix('Breed1_Name_'), how='left', left_on='Breed1_name_modified',
                  right_on='Breed1_Name_Breed_Name')
    df = pd.merge(df, df_breed_detail.add_prefix('Breed2_Name_'), how='left', left_on='Breed2_name_modified',
                  right_on='Breed2_Name_Breed_Name')
    df = df.drop('Breed2_Name_Breed_Name', axis=1)
    df = df.drop('Breed1_Name_Breed_Name', axis=1)

    df = df.drop('Breed1_name_modified', axis=1)
    df = df.drop('Breed2_name_modified', axis=1)

    for col_ in df.columns.tolist():
        if 'Breed1_Name_' in col_ or 'Breed2_Name_' in col_:
            df[col_].fillna(0, inplace=True)
    del df_breed_detail
    # Rescure count
    rescuer_count = df.groupby(['RescuerID'])['PetID'].count().reset_index()
    rescuer_count.rename(columns={rescuer_count.columns[0]: 'RescuerID'}, inplace=True)

    rescuer_count.columns = ['RescuerID', 'RescuerID_COUNT']
    df = df.merge(rescuer_count, how='left', on='RescuerID')
    df = df.drop('RescuerID', axis=1)

    Breed1_name_count = df.groupby(['Breed1_name'])['PetID'].count().reset_index()
    Breed1_name_count.rename(columns={Breed1_name_count.columns[0]: 'Breed1_name'}, inplace=True)
    Breed1_name_count.columns = ['Breed1_name', 'Breed1_name_COUNT']
    df = df.merge(Breed1_name_count, how='left', on='Breed1_name')

    Breed2_name_count = df.groupby(['Breed2_name'])['PetID'].count().reset_index()
    Breed2_name_count.rename(columns={Breed2_name_count.columns[0]: 'Breed2_name'}, inplace=True)
    Breed2_name_count.columns = ['Breed2_name', 'Breed2_name_COUNT']
    df = df.merge(Breed2_name_count, how='left', on='Breed2_name')

    df['log_count_breed1'] = (df['Breed1_name_COUNT'] + 1).apply(np.log)
    df['log_count_breed2'] = (df['Breed2_name_COUNT'] + 1).apply(np.log)

    # Description
    # TF-IDF
    df.Description.fillna('nothing', inplace=True)
    df['Description'] = df.Description.astype(str)
    df['DescLength'] = df.Description.apply(len)
    df['Lang'] = df.Description.apply(detect_language)

    df.Name.fillna('no name', inplace=True)
    # df['Name'] = df.Name.astype(str)
    df['Name'] = df['Name'].str.lower()
    df['IsNamed'] = (~df['Name'].isnull() | df['Name'].str.contains('no | none')).astype(int)
    df = df.drop('Name', axis=1)

    # Generate text features:
    print('[INFO] Generating text features')
    text_columns = ['Description', 'entities']
    for col_ in tqdm(text_columns):
        print(col_)
        text = df[col_].values.tolist()
        print('[INFO] Start count vectorize')
        cvec = CountVectorizer(min_df=5, ngram_range=(1, 3), max_features=1000,
                               strip_accents='unicode',
                               lowercase=True, analyzer='word', token_pattern=r'\w+',
                               stop_words='english')
        cvec.fit(text)
        X = cvec.transform(text)
        df['cvec_sum'] = X.sum(axis=1)
        df['cvec_mean'] = X.mean(axis=1)
        df['cvec_len'] = (X != 0).sum(axis=1)

        print('[INFO] Start TFDIDF')
        tfv = TfidfVectorizer(min_df=5, max_features=10000,
                              strip_accents='unicode', analyzer='word',
                              ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                              stop_words='english')

        # Fit TFIDF
        X = tfv.fit_transform(text)
        df['tfidf_sum'] = X.sum(axis=1)
        df['tfidf_mean'] = X.mean(axis=1)
        df['tfidf_len'] = (X != 0).sum(axis=1)
        n_components = 120

        print('[INFO] Start NMF')
        nmf_ = NMF(n_components=n_components)
        X_nmf = nmf_.fit_transform(X)
        X_nmf = pd.DataFrame(X_nmf, columns=['{}_nmf_{}'.format(col_, i) for i in range(n_components)])
        X_nmf['PetID'] = df.PetID.values.tolist()
        df = pd.concat([df.set_index('PetID'), X_nmf.set_index('PetID')], sort=False, axis=1).reset_index()
        df.rename(columns={df.columns[0]: 'PetID'}, inplace=True)

        print('[INFO] Start SVD')
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(X)
        print('fit done')
        X_svd = svd.transform(X)
        X_svd = pd.DataFrame(X_svd, columns=['{}_svd_{}'.format(col_, i) for i in range(n_components)])
        X_svd['PetID'] = df.PetID.values.tolist()
        df = pd.concat([df.set_index('PetID'), X_svd.set_index('PetID')], sort=False, axis=1).reset_index()
        df.rename(columns={df.columns[0]: 'PetID'}, inplace=True)

    for col_ in text_columns:
        df.drop(col_, axis=1, inplace=True)
    # ----------------------------------

    df['log_age'] = (df['Age'] + 1).apply(np.log)

    # Extend aggregates and improve column naming
    aggregates = ['mean', 'sum', 'var']
    columns_to_aggregate = [ 'bounding_confidence', 'bounding_importance', 'importance_mean',
                            'resolution', 'RescuerID_COUNT', 'Quantity','Age','Type']

    for col_ in columns_to_aggregate:
        if col_ != 'Breed1':
            df[col_] = df[col_].fillna(0)
            df[col_] = df[col_].astype(float)

    df['Type'] = np.where(df['Type']==1,'Dog','Cat')

    df_agg = df[columns_to_aggregate].groupby(['Age']).agg(aggregates)
    df_agg.columns = pd.Index(['{}_{}'.format(
        c[0], c[1].upper()) for c in df_agg.columns.tolist()])
    df_agg = df_agg.reset_index()
    df_agg = df_agg.fillna(0)

    df = pd.merge(df, df_agg.add_prefix('agg_'), how='left', left_on='Age', right_on='agg_Age')
    df['GrowthRate'] = df['Age'] / df['MaturitySize']
    df['WhetherVideo'] = (df['VideoAmt'] > 0).astype(int)

    df = pd.concat([df.drop('Lang', axis=1), pd.get_dummies(df['Lang'],prefix='Lang')], axis=1)
    df['isSenior'] = np.where(df['Age'] > 72, 1, 0)
    df['Number_of_media'] = df['PhotoAmt'] + 2*df['VideoAmt']

    df = extract_features(df)
    df = df.drop(['Breed1','Breed2','Age'],axis=1)
    reduce_mem_usage(df)
    return df

print('[INFO] Function defined')

# MODELISATION
def run_cv_model(df_train, df_test, target, model_fn, params={}, eval_fn=None, label='model', N_SPLITS=5):
    kf = StratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)
    fold_splits = kf.split(df_train, target)
    cv_scores = []
    qwk_scores = []
    pred_full_test = 0
    pred_train = np.zeros((df_train.shape[0], N_SPLITS))
    all_coefficients = np.zeros((N_SPLITS, 4))
    feature_importance_df = pd.DataFrame()
    i = 1
    for dev_index, val_index in fold_splits:
        print('Started ' + label + ' fold ' + str(i) + '/' + str(N_SPLITS))
        if isinstance(df_train, pd.DataFrame):
            dev_X, val_X = df_train.iloc[dev_index], df_train.iloc[val_index]
            dev_y, val_y = target[dev_index], target[val_index]
        else:
            dev_X, val_X = df_train[dev_index], df_train[val_index]
            dev_y, val_y = target[dev_index], target[val_index]
        params2 = params.copy()
        print(df_test.shape)
        pred_val_y, pred_test_y, importances, coefficients, qwk = model_fn(dev_X, dev_y, val_X, val_y, df_test, params2,
                                                                           label)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        all_coefficients[i - 1, :] = coefficients
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            qwk_scores.append(qwk)
            print(label + ' cv score {}: RMSE {} QWK {}'.format(i, cv_score, qwk))
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = df_train.columns.values
        fold_importance_df['importance'] = importances
        fold_importance_df['fold'] = i
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        i += 1
    print('{} cv RMSE scores : {}'.format(label, cv_scores))
    print('{} cv mean RMSE score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std RMSE score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv QWK scores : {}'.format(label, qwk_scores))
    print('{} cv mean QWK score : {}'.format(label, np.mean(qwk_scores)))
    print('{} cv std QWK score : {}'.format(label, np.std(qwk_scores)))
    pred_full_test = pred_full_test / float(N_SPLITS)
    results = {'label': label,
               'train': pred_train, 'test': pred_full_test,
               'cv': cv_scores, 'qwk': qwk_scores,
               'importance': feature_importance_df,
               'coefficients': all_coefficients}
    return results

def runGB(train_X, train_y, test_X, test_y, test_X2, params, mode):
    test_y = test_y.astype(int).tolist()
    print('Train GB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None
    if params.get('early_stop'):
        early_stop = params.pop('early_stop')

    if mode == 'XGB':
        d_train = xgb.DMatrix(data=train_X, label=train_y, feature_names=train_X.columns)
        d_valid = xgb.DMatrix(data=test_X, label=test_y, feature_names=test_X.columns)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(dtrain=d_train,
                          num_boost_round=num_rounds, evals=watchlist,
                          early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)

        pred_test_y = model.predict(xgb.DMatrix(test_X, feature_names=train_X.columns),
                                    ntree_limit=model.best_ntree_limit)
        pred_test_y2 = model.predict(xgb.DMatrix(test_X2, feature_names=test_X.columns),
                                     ntree_limit=model.best_ntree_limit)

    elif mode == 'LGB':
        d_train = lgb.Dataset(train_X, label=train_y)
        d_valid = lgb.Dataset(test_X, label=test_y)
        watchlist = [d_train, d_valid]
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop)
        pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
        pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
    elif mode == 'CTB':
        pass

    print('Predict 1/2')
    optR = OptimizedRounder()
    optR.fit(pred_test_y, test_y)
    coefficients = optR.coefficients()
    pred_test_y = optR.predict(pred_test_y, coefficients)
    print("Valid Counts = ", Counter(test_y))
    print("Predicted Counts = ", Counter(pred_test_y))
    print("Coefficients = ", coefficients)
    qwk = quadratic_weighted_kappa(test_y, pred_test_y)
    print("QWK = ", qwk)
    print('Predict 2/2')
    if mode == 'LGB':
        importance = model.feature_importance()
    elif mode == 'XGB':
        importance = 0
    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), importance, coefficients, qwk

def run_model(df_train, df_test):
    target = df_train['AdoptionSpeed']
    train = df_train.drop(['PetID'], axis=1)
    test_id = df_test[['PetID']]
    df_test = df_test.drop('PetID', axis=1)

    test = df_test
    categorical_features =  list(train.select_dtypes(include=['category']).columns)
    print(categorical_features)
    #train.drop(categorical_features, axis=1, inplace=True)
    #df_test.drop(categorical_features, axis=1, inplace=True)
    """
    impact_coding_map = {}
    for f in categorical_features:
        print("Impact coding for {}".format(f))
        train["impact_encoded_{}".format(f)], impact_coding_mapping, default_coding = \
                                                    impact_coding(train, f,target="AdoptionSpeed")
        impact_coding_map[f] = (impact_coding_mapping, default_coding)
        mapping, default_mean = impact_coding_map[f]
        test["impact_encoded_{}".format(f)] = test.apply(lambda x: mapping[x[f]] if x[f] in mapping
        else default_mean, axis=1)

    for cat in categorical_features:
        train = frequency_encoding(train, cat)
        test = frequency_encoding(test, cat)
    """

    train = train.drop(['AdoptionSpeed'], axis=1)

    é

    lgb_params = {'application': 'regression',
              'boosting': 'gbdt',
              'metric': 'rmse',
              'num_leaves': 70,
              'max_depth': 10,
              'learning_rate': 0.012,
              'bagging_fraction': 0.65,
              'feature_fraction': 0.8,
              'min_split_gain': 0.02,
              'min_child_samples': 100,
              'min_child_weight': 0.05,
              'lambda_l2': 0.0475,
              'verbosity': -1,
              'data_random_seed': 17,
              'early_stop': 800,
              'verbose_eval': 100,
              'num_rounds': 10000, }

    xgb_params = {
        'booster':'gbtree',
        'eval_metric': 'rmse',
        'num_leaves': 90,
        'max_depth': 300,
        'learning_rate': 0.01,
        'bagging_fraction': 0.85,
        'feature_fraction': 0.8,
        'min_split_gain': 0.02,
        'min_child_samples':100,
        'min_child_weight': 0.02,
        'lambda_l2': 0.0475,
        'verbose_eval': 1,
        'data_random_seed': 17,
        'early_stop': 900,
        'num_rounds': 10000
 }
    #'tree_method': 'gpu_hist',
    #s'device': 'gpu',

    cat_params = {
    }

    results = run_cv_model(train, test, target, runGB, lgb_params, rmse, 'LGB', N_SPLITS)
    imports = results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()
    imports.reset_index(inplace=True,drop=False)
    imports = imports.sort_values('importance', ascending=False)
    print(imports)
    optR = OptimizedRounder()
    coefficients_ = np.mean(results['coefficients'], axis=0)
    print(coefficients_)
    #manually adjust coefs
    #coefficients_[0] = 1.65   #1.66
    #coefficients_[1] = 2.15   #2.13
    #coefficients_[2] = 2.60
    #coefficients_[3] = 2.8
    train_predictions = [r[0] for r in results['train']]
    train_predictions = optR.predict(train_predictions, coefficients_).astype(int)

    test_predictions = [r[0] for r in results['test']]
    test_predictions = optR.predict(test_predictions, coefficients_).astype(int)

    print(Counter(train_predictions), Counter(test_predictions))
    print('[RESULT] ON TRAINING SET')
    print('QWK {}', format(quadratic_weighted_kappa(target.astype(int).tolist(), train_predictions)))
    print('RMSE {}', format(rmse(target, [r[0] for r in results['train']])))

    print("True Distribution:")
    print(pd.value_counts(target, normalize=True).sort_index())
    print("Test Predicted Distribution:")
    print(pd.value_counts(test_predictions, normalize=True).sort_index())
    print("Train Predicted Distribution:")
    print(pd.value_counts(train_predictions, normalize=True).sort_index())
    pd.DataFrame(sk_cmatrix(target, train_predictions), index=list(range(5)), columns=list(range(5)))
    submission = pd.DataFrame({'PetID': test_id['PetID'].tolist(), 'AdoptionSpeed': test_predictions})
    submission.head()
    submission.to_csv('submission.csv', index=False)

def get_data_with_features_engineering():
    plt.rcParams['figure.figsize'] = (12, 9)
    plt.style.use('ggplot')

    debug = 0
    sample = 1000
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)

    if debug:
        df_train = df_train.head(sample)
        df_test = df_test.head(sample)

    df = features_engineering([df_train, df_test])
    print(df.shape)
    return df

if __name__ =='__main__':
    global_file = 1
    if global_file:
        df = pd.read_csv('global.csv',header=0, sep=";")
        df = reduce_mem_usage(df)
    else:
        df = get_data_with_features_engineering()
    df_train = df[~df.AdoptionSpeed.isnull()]
    df_test = df[df.AdoptionSpeed.isnull()]
    df_test.drop('AdoptionSpeed', axis=1, inplace=True)
    df.to_csv('global.csv', sep=";", header=True, index=False)
    del df
    run_model(df_train, df_test)
