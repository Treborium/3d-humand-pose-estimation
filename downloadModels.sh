# Create an folder
mkdir models

# Download current models
wget http://gvv.mpi-inf.mpg.de/projects/XNect/assets/models/SelecSLS42_B_statedict.pth -O models/SLS42_B.pth
wget http://gvv.mpi-inf.mpg.de/projects/XNect/assets/models/SelecSLS60_statedict_better.pth -O models/SLS60.pth
wget http://gvv.mpi-inf.mpg.de/projects/XNect/assets/models/SelecSLS60_B_statedict.pth -O models/SLS60_B.pth

# Download old models
wget http://gvv.mpi-inf.mpg.de/projects/XNect/assets/models/SelecSLS60_statedict.pth -O models/SLS60_old.pth
wget http://gvv.mpi-inf.mpg.de/projects/XNect/assets/models/SelecSLS84_statedict.pth -O models/SLS84_old.pth