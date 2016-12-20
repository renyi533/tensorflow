package(default_visibility=["//visibility:public"])
licenses(["notice"])

cc_library(
    name = "open_blas",
    srcs = ["lib/libopenblas.so"],
    hdrs = ["include/cblas.h", "include/f77blas.h", "include/openblas_config.h"],
    copts = ["-Ithird_party/open_blas"],
    )
    
