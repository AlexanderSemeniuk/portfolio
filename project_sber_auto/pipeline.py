from datetime import datetime
import dill
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict


def fillna(df):
    df_fill = df.copy()
    columns_to_fill = [
        'utm_adcontent',
        'utm_campaign',
        'utm_source'
    ]

    for column in columns_to_fill:

        df_fill[column] = df_fill[column].fillna(df_fill[column].mode()[0])

    return df_fill

def none_not_set(df):
    df_none_not_set = df.copy()
    columns_none_not_set  = [
        'utm_medium',
        'device_os',
        'device_brand',
        'device_browser',
        'geo_country',
        'geo_city'
    ]
    for column in columns_none_not_set:

        df_none_not_set.loc[(df_none_not_set[column] ==  '(not set)') |
                            (df_none_not_set[column] == '(none)'), [column]] = df_none_not_set[column].mode()[0]

    return df_none_not_set

def device_os_brand(df):
    df_dev_os = df.copy()
    df_dev_os.device_os = df_dev_os.device_os.fillna(df_dev_os.device_brand.apply(lambda x: "iOS" if x == 'Apple'
                                  else ("Windows" if x == 'Microsoft'
                                  else ('Android' if x in ['Huawei', 'Samsung',
                                   'Xiaomi', 'Vivo', 'Meizu', 'OnePlus',
                                   'Realme', 'OPPO', 'Infinix',
                                   'Micromax', 'Blackview', 'Oukitel',
                                   'Wileyfox', 'Motorola', 'HOMTOM',
                                   'Cubot', 'DOOGEE', 'ZTE']
                                  else ("Other")))))
    df_dev_os.device_brand = df_dev_os.device_brand.fillna(df_dev_os.device_os.apply(lambda x:
                                  'Apple' if x == "iOS" else ('Microsoft' if x == "Windows" else ("Other"))))

    return df_dev_os

def date_feature(df):

    df_day = df.copy()
    df_day['visit_date'] = pd.to_datetime(df_day['visit_date'], utc=True)
    df_day['day'] = df_day['visit_date'].dt.day
    df_day['dayofweek'] = df_day['visit_date'].dt.weekday
    df_day['mounth'] = df_day['visit_date'].dt.month
    df_day['hour'] = df_day['visit_time'].apply(lambda x: x.lower().split(':')[0])
    df_day['day_time'] = df_day['hour'].apply(lambda x: 0 if x in ['21', '22', '23', '00', '01', '02', '03']
    else (1 if x in ['03', '04', '05', '06', '07', '08', '09']
          else (2 if x in ['09', '10', '11', '12', '13', '14', '15']
                else (3))))

    return df_day

def screen_res(df):

    df_screen_res = df.copy()
    df_screen_res['device_screen_resolution'] = df_screen_res['device_screen_resolution'].apply(
        lambda x: int(x.lower().split('x')[0]) * int(x.lower().split('x')[1]))

    return df_screen_res

def dev_cat_new(df):

    df_dev_cat_new = df.copy()
    le = preprocessing.LabelEncoder()
    le.fit(df_dev_cat_new.device_category)
    le.transform(df_dev_cat_new.device_category)
    df_dev_cat_new['device_category_new'] = pd.DataFrame(le.transform(df_dev_cat_new.device_category))

    return df_dev_cat_new


def geo_city_new(df):

    df_geo_city_new = df.copy()
    df_geo_city_new['geo_country'] = df_geo_city_new.apply(lambda x: 1 if x.geo_country == 'Russia' else 0, axis=1)
    df_geo_city_new['geo_city_new'] = df_geo_city_new.geo_city.apply(lambda x: 1 if x == "Moscow"
    else (2 if x == "Saint Petersburg"
          else (3 if x == "Yekaterinburg"
                else (4 if x == "Krasnodar"
                      else (0)))))

    return df_geo_city_new

def main():

    df = pd.read_csv('data/df_pip_dropdupl.csv')

    X = df.drop(columns = ['target', 'device_model', 'utm_keyword', 'session_id', 'client_id', 'visit_number'], axis=1)
    y = df['target']

    numerical_feat = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_feat = X.select_dtypes(include=['object']).columns

    preprocessor_func = Pipeline(steps=[
        ('date_feature', FunctionTransformer(date_feature)),
        ('screen_res', FunctionTransformer(screen_res)),
        ('dev_cat_new', FunctionTransformer(dev_cat_new)),
        ('fillna', FunctionTransformer(fillna)),
        ('geo_city_new', FunctionTransformer(geo_city_new)),
        ('none_not_set', FunctionTransformer(none_not_set)),
        ('device_os_brand', FunctionTransformer(device_os_brand))
            ])

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', min_frequency=1782, sparse_output=True))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_feat),
        ('categorical', categorical_transformer, categorical_feat),

    ])

    preprocessor = Pipeline(steps=[
        ('column_transformer', column_transformer)
    ])

    models = (
        MLPClassifier(),
        MLPClassifier(activation='logistic', hidden_layer_sizes=(256, 128, 64))
           )

    best_score = .0
    best_pipe = None

    for model in models:

        pipe = Pipeline(steps=[
            ('preprocessor_func', preprocessor_func),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')
        print(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc_std: {best_score:.4f}')

    best_pipe.fit(X, y)
    with open('model/model.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Target prediction model',
                'author': 'Alexandr Semeniuk',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'roc_auc': best_score
            }
        }, file, recurse=True)

if __name__ == '__main__':
    main()
