import utils


def main():
    print('Creating the test dataset...')
    df = utils.build_dataset(wav_number=50, random_sate=43)

    df['prediction'] = df.apply(utils.model2, axis=1)
    acc = utils.accuracy(df['speaker'], df["prediction"])

    print(f'The cepstrum-pitch-based model has {acc*100:.2f}% accuracy.')


if __name__ == '__main__':
    main()
