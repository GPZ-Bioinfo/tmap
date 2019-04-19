git tag 1.3
git push --tags
git add .
git commit -m 'Improve the performance of SAFE'
git push origin -u master
python3 setup.py sdist upload -r pypi
