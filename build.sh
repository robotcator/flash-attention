
#rm -rf build flash_attn_cuda.cpython-37m-x86_64-linux-gnu.so

start=`date +%s`
#CXX="/usr/lib/ccache/c++" 
python setup.py build -j 8 develop 2>&1 | tee build.log
end=`date +%s`

runtime=$((end-start))
echo ${runtime}
