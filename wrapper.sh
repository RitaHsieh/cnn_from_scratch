#! /bin/bash

mkdir -p nsys_reports

nsys profile \
-o "./nsys_reports/profile.nsys-rep" \
--mpi-impl mpich \
--trace cuda,mpi,osrt \
$@

