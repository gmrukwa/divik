import logging

from divik.feature_extraction import LocallyAdjustedRbfSpectralEmbedding


def main():
    import divik._cli._utils as scr
    data, config, destination, _ = scr.initialize()
    try:
        spectral = LocallyAdjustedRbfSpectralEmbedding(**config)
        spectral.fit(data)
        spectral.save(destination)
    except Exception as ex:
        logging.error("Failed with exception.")
        logging.error(repr(ex))
        raise


if __name__ == '__main__':
    main()
