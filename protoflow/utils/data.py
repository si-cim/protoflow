"""ProtoFlow data utilities."""

import hashlib
import os
import shutil
import tarfile
import zipfile

import requests
import six
from six.moves.urllib.error import HTTPError, URLError
from tensorflow.python.keras.utils.io_utils import path_to_string
from tqdm import tqdm


def _extract_archive(file_path, path=".", archive_format="auto"):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

  Arguments:
      file_path: path to the archive file
      path: path to extract the archive file
      archive_format: Archive format to try for extracting the file.
          Options are "auto", "tar", "zip", and None.
          "tar" includes tar, tar.gz, and tar.bz files.
          The default "auto" is ["tar", "zip"].
          None or an empty list will return no matches found.

  Returns:
      True if a match was found and an archive extraction was completed,
      False otherwise.
  """
    if archive_format is None:
        return False
    if archive_format == "auto":
        archive_format = ["tar", "zip"]
    if isinstance(archive_format, six.string_types):
        archive_format = [archive_format]

    file_path = path_to_string(file_path)
    path = path_to_string(path)

    for archive_type in archive_format:
        if archive_type == "tar":
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == "zip":
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False


def _hash_file(fpath, algorithm="sha256", chunk_size=65535):
    """Calculates a file sha256 or md5 hash.

  Example:

  ```python
  _hash_file("/path/to/file.zip")
  "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  ```

  Arguments:
      fpath: path to the file being validated
      algorithm: hash algorithm, one of `"auto"`, `"sha256"`, or `"md5"`.
          The default `"auto"` detects the hash algorithm in use.
      chunk_size: Bytes to read at a time, important for large files.

  Returns:
      The file hash
  """
    if (algorithm == "sha256") or (algorithm == "auto" and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, "rb") as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def _validate_file(fpath, file_hash, algorithm="auto", chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.

  Arguments:
      fpath: path to the file being validated
      file_hash:  The expected hash string of the file.
          The sha256 and md5 hash algorithms are both supported.
      algorithm: Hash algorithm, one of "auto", "sha256", or "md5".
          The default "auto" detects the hash algorithm in use.
      chunk_size: Bytes to read at a time, important for large files.

  Returns:
      Whether the file is valid
  """
    if (algorithm == "sha256") or (algorithm == "auto"
                                   and len(file_hash) == 64):
        hasher = "sha256"
    else:
        hasher = "md5"

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def _save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        pbar = tqdm(total=None)
        progress = 0
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress += len(chunk)
                pbar.update(progress - pbar.n)
        pbar.close()


def get_file_from_google(fname,
                         file_id,
                         untar=False,
                         md5_hash=None,
                         file_hash=None,
                         cache_subdir="datasets",
                         hash_algorithm="auto",
                         extract=False,
                         archive_format="auto",
                         cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".keras")
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = "md5"
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join("/tmp", ".keras")
    datadir = os.path.join(datadir_base, cache_subdir)
    os.makedirs(datadir, exist_ok=True)

    fname = path_to_string(fname)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + ".tar.gz"
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not _validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print("A local file was found, but it seems to be "
                      "incomplete or outdated because the " + hash_algorithm +
                      " file hash does not match the original value of " +
                      file_hash + " so we will re-download the data.")
                download = True
    else:
        download = True

    if download:
        print("Downloading data from Google Drive...")

        error_msg = "Failed on https://drive.google.com/file/d/{}: {} -- {}"
        try:
            try:
                url = "https://docs.google.com/uc?export=download"
                session = requests.Session()

                response = session.get(url,
                                       params={"id": file_id},
                                       stream=True)
                token = _get_confirm_token(response)

                if token:
                    params = {"id": file_id, "confirm": token}
                    response = session.get(url, params=params, stream=True)

                _save_response_content(response, fpath)
            except HTTPError as e:
                raise Exception(error_msg.format(file_id, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(file_id, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise e

    if untar:
        if not os.path.exists(untar_fpath):
            _extract_archive(fpath, datadir, archive_format="tar")
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath
