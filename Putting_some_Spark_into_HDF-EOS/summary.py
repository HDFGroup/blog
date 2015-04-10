import os, sys

import h5py, numpy as np

from pyspark import SparkContext

if __name__ == "__main__":
    """
    Usage: doit [partitions]
    """
    sc = SparkContext(appName="GSSTF_NCEP.3")
    base_dir = str(sys.argv[1]) if len(sys.argv) > 1 else exit(-1)
    partitions = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    # change this if you want a different dataset
    hdf5_path = "/HDFEOS/GRIDS/NCEP/Data Fields/Tair_2m"

    ###################################################################
    ### get the sample count, mean, median, and standard deviation

    def summarize(x):
        a = x.split(",")
        file_name = a[0]
        h5path = a[1].strip()

        with h5py.File(file_name) as f:
            dset = f[h5path]
            # Spark doesn't like dset[...]. Why?
            tair_2m = dset[:,:]
            fill = dset.attrs['_FillValue'][0]
            # mask fill values
            v = tair_2m[tair_2m != fill]
            # extract the date from the file name
            # GSSTF_NCEP.3.YYYY.MM.DD.he5
            key = "".join(file_name[-14:-4].split("."))
            return [(key, (len(v), np.mean(v), np.median(v), np.std(v)))]

    ###################################################################

    # traverse the base directory and pick up HDFEOS files
    file_list = filter(lambda s: s.endswith(".he5"),
                       [ "%s%s%s" % (root, os.sep, file)
                         for root, dirs, files in os.walk(base_dir)
                         for file in files])

    # partition the list
    file_paths = sc.parallelize(
        map(lambda s: "%s,%s"%(s, hdf5_path), file_list), partitions)

    # compute per file summaries
    rdd = file_paths.flatMap(summarize)

    # collect the results and write the time series to a CSV file
    results = rdd.collect()

    with open("summary.csv", "w") as text_file:
        text_file.write("Day,Samples,Mean,Median,Stdev\n")
        for k in sorted(results):
            text_file.write(
                "{0},{1},{2},{3},{4}\n".format(
                    k[0], k[1][0], k[1][1], k[1][2], k[1][3]))

    sc.stop()
