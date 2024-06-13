import joblib
from typing import Tuple, Dict
import pandas as pd


class GenderPredictor:

    def __init__(self):

        self.predictor_names = joblib.load('models/predictor_names.pkl')
        self.names_vectorizer = joblib.load('models/names_vectorizer.pkl')
        self.predictor_surnames = joblib.load('models/predictor_surnames.pkl')
        self.surnames_vectorizer = joblib.load('models/surnames_vectorizer.pkl')
        self.predictor_patronymics = joblib.load('models/predictor_patronymics.pkl')
        self.patronymics_vectorizer = joblib.load('models/patronymics_vectorizer.pkl')

    def predict_gender_by_full_name(
            self,
            surname: str = None,
            name: str = None,
            patronymic: str = None
    ) -> Tuple[str, Dict]:

        if not surname and not name and not patronymic:
            raise ValueError('All parameters are none')
        probabilities = {'М': [], 'Ж': []}
        if surname:
            surname_cleaned = surname.lower().replace('ё', 'е')
            surname_vector = self.surnames_vectorizer.transform([surname_cleaned])
            surname_probabilities = self.predictor_surnames.predict_proba(surname_vector)[0]
            surname_predictions = {class_: proba for class_, proba in
                                   zip(self.predictor_surnames.classes_, surname_probabilities)}
            if max(surname_predictions, key=surname_predictions.get) != 'Н':
                probabilities['М'].append(surname_predictions['М'])
                probabilities['Ж'].append(surname_predictions['Ж'])
        if name:
            name_cleaned = name.lower().replace('ё', 'е')
            name_vector = self.names_vectorizer.transform([name_cleaned])
            name_probabilities = self.predictor_names.predict_proba(name_vector)[0]
            name_predictions = {class_: proba for class_, proba
                                in zip(self.predictor_names.classes_, name_probabilities)}
            probabilities['М'].append(name_predictions['М'])
            probabilities['Ж'].append(name_predictions['Ж'])
        if patronymic:
            patronymic_cleaned = patronymic.lower().replace('ё', 'е')
            patronymic_vector = self.patronymics_vectorizer.transform([patronymic_cleaned])
            patronymic_probabilities = self.predictor_patronymics.predict_proba(patronymic_vector)[0]
            patronymic_predictions = {class_: proba for class_, proba in
                                      zip(self.predictor_patronymics.classes_, patronymic_probabilities)}
            probabilities['М'].append(patronymic_predictions['М'])
            probabilities['Ж'].append(patronymic_predictions['Ж'])
        predictions = {gender: sum(probability) / len(probability) for gender, probability in probabilities.items()}
        return max(predictions, key=predictions.get), predictions

    def predict_gender_for_dataframe(
            self,
            df: pd.DataFrame,
            surname_label: str = 'surname',
            name_label: str = 'name',
            patronymic_label: str = 'patronymic',
            gender_label='gender',
            inplace: bool = True
    ) -> pd.DataFrame:

        if surname_label in df.columns:
            surnames = df.loc[~df[surname_label].isnull(), surname_label].str.lower().str.replace('ё', 'е')
            surname_predictions = pd.DataFrame(
                self.predictor_surnames.predict_proba(self.surnames_vectorizer.transform(surnames)),
                columns=self.predictor_surnames.classes_,
                index=surnames.index
            )
            surname_predictions = surname_predictions.loc[surname_predictions['Н'] != surname_predictions.max(axis=1)]
        else:
            surname_predictions = pd.DataFrame([], columns=self.predictor_surnames.classes_)
        if name_label in df.columns:
            names = df.loc[~df[name_label].isnull(), name_label].str.lower().str.replace('ё', 'е')
            name_predictions = pd.DataFrame(
                self.predictor_names.predict_proba(self.names_vectorizer.transform(names)),
                columns=self.predictor_names.classes_,
                index=names.index
            )
        else:
            name_predictions = pd.DataFrame([], columns=self.predictor_names.classes_)
        if patronymic_label in df.columns:
            patronymics = df.loc[~df[patronymic_label].isnull(), patronymic_label].str.lower().str.replace('ё', 'е')
            patronymic_predictions = pd.DataFrame(
                self.predictor_patronymics.predict_proba(self.patronymics_vectorizer.transform(patronymics)),
                columns=self.predictor_patronymics.classes_,
                index=patronymics.index
            )
        else:
            patronymic_predictions = pd.DataFrame([], columns=self.predictor_patronymics.classes_)
        predictions = surname_predictions.join(name_predictions, how='outer', rsuffix='_name').join(
            patronymic_predictions, how='outer', lsuffix='_surname', rsuffix='_patronymic')
        predictions['М'] = predictions[['М_surname', 'М_name', 'М_patronymic']].mean(axis=1)
        predictions['Ж'] = predictions[['Ж_surname', 'Ж_name', 'Ж_patronymic']].mean(axis=1)
        if inplace:
            df[gender_label] = pd.Series(dtype=str)
            df.loc[predictions[predictions['М'] >= predictions['Ж']].index, gender_label] = 'М'
            df.loc[predictions[predictions['М'] < predictions['Ж']].index, gender_label] = 'Ж'
            return None
        else:
            new_df = df.copy()
            new_df[gender_label] = pd.Series(dtype=str)
            new_df.loc[predictions[predictions['М'] >= predictions['Ж']].index, gender_label] = 'М'
            new_df.loc[predictions[predictions['М'] < predictions['Ж']].index, gender_label] = 'Ж'
            return new_df
