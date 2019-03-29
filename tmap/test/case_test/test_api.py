import os
from subprocess import check_call

from tmap.api.general import randomString

itspath = os.path.abspath(__file__)
data_dir = os.path.join(os.path.dirname(itspath), 'test_data')
api_dir = os.path.join(os.path.dirname(itspath), '..', 'api')
out_dir = "/tmp/tmp_%s" % randomString(5)

p_path = '/usr/bin/python3'
cmdlines = """
{ppath} {api}/envfit_analysis.py -I {data_dir}/FGFP_genus_data.csv -M {data_dir}/FGFP_metadata.tsv -O {odir}/FGFP_envfit.csv -tn temp --keep -v
{ppath} {api}/Network_generator.py -I {data_dir}/FGFP_genus_data.csv -O {odir}/FGFP.graph -v
{ppath} {api}/SAFE_analysis.py both -G {odir}/FGFP.graph -M {odir}/temp.envfit.data {odir}/temp.envfit.metadata -P {odir}/FGFP -i 1000 -p 0.05 -r -v
{ppath} {api}/SAFE_visualization.py ranking -G {odir}/FGFP.graph -S1 {odir}/FGFP_raw_enrich -S2 {odir}/FGFP_temp.envfit.data_enrich.csv {odir}/FGFP_temp.envfit.metadata_enrich.csv -O {odir}/FGFP_ranking.html
{ppath} {api}/SAFE_visualization.py stratification -G {odir}/FGFP.graph -S1 {odir}/FGFP_raw_enrich -S2 {odir}/FGFP_temp.envfit.data_enrich.csv {odir}/FGFP_temp.envfit.metadata_enrich.csv -O {odir}/FGFP_stratification.html
{ppath} {api}/SAFE_visualization.py ordination -G {odir}/FGFP.graph -S1 {odir}/FGFP_raw_enrich -S2 {odir}/FGFP_temp.envfit.data_enrich.csv {odir}/FGFP_temp.envfit.metadata_enrich.csv -O {odir}/FGFP_ordination.html
""".format(ppath=p_path,
           data_dir=data_dir,
           api=api_dir,
           odir=out_dir)

cmdlines = [_ for _ in cmdlines.split('\n') if _]

cmdlines += ['rm -r %s' % out_dir]

if __name__ == '__main__':
    for _ in cmdlines:
        try:
            check_call([_ for _ in _.split(' ') if _])
        except Exception as e:
            print(e)
