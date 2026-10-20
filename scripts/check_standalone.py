"""Copy only this release to /tmp and check it without the original project on sys.path."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import SimpleITK as sitk


def main():
    source = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='pgrmamba_standalone_') as temp:
        isolated = Path(temp) / 'PGRMamba'
        shutil.copytree(source, isolated, ignore=shutil.ignore_patterns(
            '3D_NSD', '__pycache__', '.cache', 'runs', 'outputs', 'build', 'dist', '*.egg-info'))
        # Small independent fixtures keep this check fast and do not require copying the dataset.
        for split in ('train', 'val'):
            directory = isolated / '3D_NSD' / split
            directory.mkdir(parents=True)
            sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32)),
                           str(directory / 'B1_001.tif'))
            (directory / 'B1_001.csv').write_text('2,3,4\n')
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        wrapper = (
            "import sys,runpy; root=sys.argv.pop(1); sys.path.insert(0,root); "
            "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
        )
        for entry, args in [('train_detction_um.py', ['--help']),
                            ('predict.py', ['--help']), ('evaluate.py', ['--help']),
                            ('evaluate_prf1_csv_um.py', ['--help']),
                            ('scripts/check_protocol.py', [])]:
            result = subprocess.run([sys.executable, '-I', '-B', '-c', wrapper, str(isolated),
                                     str(isolated / entry), *args], cwd=temp, env=env,
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{entry} failed:\n{result.stdout}')
            print(f'PASS isolated entry: {entry}', flush=True)
        imports = '''import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import load_data, get_loss, evaluation, train_detction_um, inference, predict, evaluate, evaluate_prf1_csv_um
import feature_extract_net, mamba.mamba, mamba_ssm
root = Path(sys.argv[1]).resolve()
for module in (load_data, get_loss, evaluation, train_detction_um, inference, predict, evaluate, evaluate_prf1_csv_um,
               feature_extract_net, mamba.mamba, mamba_ssm):
    assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
model = feature_extract_net.Model_net([3, 3])
assert model.mamba.ehance.fusion_channels == 8
assert sum(p.numel() for p in model.parameters()) > 0
print('All runtime project modules resolve inside the copied release.')
'''
        subprocess.run([sys.executable, '-I', '-B', '-c', imports, str(isolated)],
                       cwd=temp, env=env, check=True)
        print('PASS: copied release is independent of its original parent project.')


if __name__ == '__main__':
    main()
