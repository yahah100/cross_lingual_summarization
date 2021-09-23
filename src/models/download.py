# Download the models from Google Drive
from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1--mMs6j6z6g9qLxj69B3j5etQgkto5pK',
                                    dest_path='./t5_sueddeutsche.zip',
                                    unzip=True)

gdd.download_file_from_google_drive(file_id='1-DIk0OIKkzhVPP8j2fFcBXxAaSFlH215',
                                    dest_path='./t5_sueddeutsche_plus_translation.zip',
                                    unzip=True)

gdd.download_file_from_google_drive(file_id='1-2VSZwpToAHNulvaeEGP_dC0Ic_VZRnY',
                                    dest_path='./t5_cnn_daily_mail.zip',
                                    unzip=True)

gdd.download_file_from_google_drive(file_id='1-A0KgnwoyEB-du7v4XHC4n4BrbHXz2S-',
                                    dest_path='./t5_cnn_daily_mail_plus_translation.zip',
                                    unzip=True)
