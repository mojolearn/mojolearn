import json, sys, urllib.request
import torch
major, minor = torch.__version__.split('.')[:2]
cuda = torch.version.cuda.split('.')[0]
abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
py = 'cp%d%d' % sys.version_info[:2]
needle = '+cu%storch%s.%scxx11abi%s-%s-%s-linux_x86_64.whl' % (cuda, major, minor, abi, py, py)
# This official release supplies the torch2.4/Python3.11 image's prebuilt
# selective_scan_cuda. Refuse another ABI; never attempt an sdist build.
req = urllib.request.Request('https://api.github.com/repos/state-spaces/mamba/releases/tags/v2.2.4',
                             headers={'Accept': 'application/vnd.github+json'})
with urllib.request.urlopen(req, timeout=20) as response:
    release = json.load(response)
matches = [a['browser_download_url'] for a in release['assets'] if needle in a['name']]
if len(matches) != 1:
    raise SystemExit('No unique prebuilt mamba wheel for ' + needle)
print(matches[0])
