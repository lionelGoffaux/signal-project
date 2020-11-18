import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():
    print('Creating the dataset...')
    df = utils.build_dataset(wav_number=50, random_sate=43)

    df['prediction'] = df.apply(utils.model2, axis=1)
    acc = utils.accuracy(df['speaker'], df["prediction"])

    print(f'The cepstrum-pitch-based model has {acc*100:.2f}% accuracy.')

    df = df.drop(['fs', 'duration', 'prediction'], axis=1)
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    X_train, y_train = utils.preprocessing(train_set)
    X_test, y_test = utils.preprocessing(test_set)


    rforest = RandomForestClassifier(random_state=42)
    rforest.fit(X_train, y_train)
    ypred = rforest.predict(X_test)

    acc = utils.accuracy(y_test, ypred)
    print(f'The machine learning based model has {acc * 100:.2f}% accuracy.')


if __name__ == '__main__':
    main()
