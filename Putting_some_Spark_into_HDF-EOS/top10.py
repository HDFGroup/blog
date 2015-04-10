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
    ### get the bottom and top 10 temperatures

    def top10(x):
        a = x.split(",")
        file_name = a[0]
        h5path = a[1].strip()

        with h5py.File(file_name) as f:
            dset = f[h5path]
            cols = dset.shape[1]
            # Spark doesn't like dset[...]. Why?
            tair_2m = np.ravel(dset[:,:])
            pos = np.argsort(tair_2m)
            # offset the minimum by fill values count
            fill = dset.attrs['_FillValue'][0]
            offset = len(tair_2m[tair_2m == fill])
            results = []
            # bottom 10
            for p in pos[offset:offset+10]:
                results.append((p/cols, p%cols, tair_2m[p]))
            # top 10
            for p in pos[-11:-1]:
                results.append((p/cols, p%cols, tair_2m[p]))
            # extract the date from the file name
            # GSSTF_NCEP.3.YYYY.MM.DD.he5
            key = "".join(file_name[-14:-4].split("."))
            return [(key, results)]

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
    rdd = file_paths.flatMap(top10)

    # collect the results and write the time series to a CSV file
    results = rdd.collect()

    with open("top10.csv", "w") as text_file:
        text_file.write("Day")
        for i in range(1,21):
            text_file.write(",Lat{0},Lon{1},T{2}".format(i, i, i))
        text_file.write("\n")
        for k in sorted(results):
            text_file.write("{0}".format(k[0]))
            for i in range(len(k[1])):
                text_file.write(
                    ",{0},{1},{2}".format(
                        k[1][i][0], k[1][i][1], k[1][i][2]))
            text_file.write("\n")

    sc.stop()
