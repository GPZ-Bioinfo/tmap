git tag 1.3
git push --tags
git add .
git commit -m 'Improve the performance of SAFE, fix some bugs of filter and plot'
git push origin -u master
python3 setup.py sdist upload -r pypi
