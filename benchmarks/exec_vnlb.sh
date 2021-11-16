mkdir video_nlbayes3d_release
cd video_nlbayes3d_release/
git clone git@github.com:pariasm/vnlb.git .
mkdir build
cd build/
cmake ..
make
cd bin/
mkdir -p data/mobile-gray
cd data/
wget http://dev.ipol.im/~pariasm/video_nlbayes/videos/gmobile.avi . 
ffmpeg -i gmobile.avi -f image2 mobile-gray/%03d.png
cd ..
export OMP_NUM_THREADS=4
./vnlb-gt.sh data/mobile-gray/%03d.png 1 300 20 7x2/mobile_mono-s20/ "-px1 7 -pt1 2 -px2 7 -pt2 2"
grep Total 7x2/mobile_gray-s20/measures-deno
