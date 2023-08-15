from  transformers.modeling_utils import  is_accelerate_available,is_bitsandbytes_available
from transformers.utils.import_utils import _is_package_available,_bitsandbytes_available

print(is_bitsandbytes_available()==_is_package_available("bitsandbytes"))
print(_is_package_available("bitsandbytes"))
print(_is_package_available("bitsandbytes"))

import importlib.metadata
import importlib.util

pkg_name='bitsandbytes'
package_exists = importlib.util.find_spec(pkg_name) is not None
package_version = "N/A"
if package_exists:
    try:
        package_version = importlib.metadata.version(pkg_name)
        package_exists = True
    except importlib.metadata.PackageNotFoundError:
        package_exists = False
    print(f"Detected {pkg_name} version {package_version}")

