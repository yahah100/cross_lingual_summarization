# Download datasets from Google Drive
from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1vs6-g1B2ucIFcmNvAFZU0GiinBXOthU_',
                                    dest_path='./sueddeutsche.zip',
                                    unzip=True)

gdd.download_file_from_google_drive(file_id='16DhTEYKsTH9fEglD3mk7MnznHcQPugBZ',
                                    dest_path='./cnn_daily_mail.zip',
                                    unzip=True)