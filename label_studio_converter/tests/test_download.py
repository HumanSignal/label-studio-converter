from .utils import download
import os
from pathlib import Path

def test_local_download(tmp_path):
    """
    Test local download
    """
    test_file = tmp_path / "app/Highway.jpg"
    # create a directory "app/local" in temp folder
    test_file.parent.mkdir()

    url = '/data/upload/10/Highway.jpg'
    output_dir = tmp_path / 'app/images/'
    filename = None
    project_dir = tmp_path / "app/"
    return_relative_path = True
    upload_dir = tmp_path / "app/"
    download_resources = False
    f = download(url=url,
                 output_dir=output_dir,
                 filename=filename,
                 project_dir=project_dir,
                 return_relative_path=return_relative_path,
                 upload_dir=upload_dir,
                 download_resources=download_resources)
    assert f == str(Path('images/Highway.jpg'))


def test_local_serving_download(tmp_path):
    """
    Test local serving
    """
    test_file = tmp_path / "app/Highway.jpg"
    # create a directory "app/local" in temp folder
    test_file.parent.mkdir()
    # create temp file
    test_file.touch()
    url = '/data/Highway.jpg?d=' + str(test_file.parent)
    output_dir = 'app/images/'
    filename = None
    project_dir = tmp_path / "app/"
    return_relative_path = False
    upload_dir = tmp_path / "app/"
    download_resources = False
    f = download(url=url,
                 output_dir=output_dir,
                 filename=filename,
                 project_dir=project_dir,
                 return_relative_path=return_relative_path,
                 upload_dir=upload_dir,
                 download_resources=download_resources)
    assert f == str(test_file)


def test_external_download_relative(tmp_path):
    """
    Test external download
    """
    test_file = tmp_path / "app/test.jpg"
    # create a directory "app/local" in temp folder
    test_file.parent.mkdir()
    url = 'https://htx-pub.s3.amazonaws.com/datasets/images/120737490_161331265622999_4627055643295174519_n.jpg'
    output_dir = test_file.parent
    filename = None
    project_dir = tmp_path / "app/"
    return_relative_path = True
    upload_dir = tmp_path / "app/"
    download_resources = True
    f = download(url=url,
                 output_dir=output_dir,
                 filename=filename,
                 project_dir=project_dir,
                 return_relative_path=return_relative_path,
                 upload_dir=upload_dir,
                 download_resources=download_resources)
    assert 'app' in f and '120737490_161331265622999_4627055643295174519_n' in f


def test_external_download_not_relative(tmp_path):
    """
    Test external download
    """
    test_file = tmp_path / "app/test.jpg"
    # create a directory "app/local" in temp folder
    test_file.parent.mkdir()
    url = 'https://htx-pub.s3.amazonaws.com/datasets/images/120737490_161331265622999_4627055643295174519_n.jpg'
    output_dir = test_file.parent
    filename = None
    project_dir = None
    return_relative_path = False
    upload_dir = tmp_path / "app/"
    download_resources = True
    f = download(url=url,
                 output_dir=output_dir,
                 filename=filename,
                 project_dir=project_dir,
                 return_relative_path=return_relative_path,
                 upload_dir=upload_dir,
                 download_resources=download_resources)
    assert os.path.join(output_dir, os.path.basename(url).split(".")[0]) in f