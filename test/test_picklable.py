import io
import pickle
import divik.cluster as ctr


def test_configurable_classes_are_picklable():
    obj = ctr.DiviK(kmeans=ctr.GAPSearch(kmeans=ctr.KMeans(n_clusters=2), max_clusters=2))
    with io.BytesIO() as f:
        pickle.dump(obj, f)
        f.seek(0)
        obj2 = pickle.load(f)
    assert str(obj) == str(obj2)  # Should work, as involved classes inherit from BaseEstimator
