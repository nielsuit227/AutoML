from Amplo.API.platform import PlatformSynchronizer
from Amplo.Utils.testing import make_production_data
from tests.Unit.API import APITestCase


class TestPlatform(APITestCase):

    def test_upload(self):
        """
        This only tests that upload_latest_model doesn't error.
        - Good to check whether the file exists in the cloud.
        - Also needs some cleanup to make sure it doesn't stay there.
        - Also does upload_latest_model truly take the latest?
        - What if latest locally and in the cloud are outdated? It needs to go in the right folder.
        """

        # # Make dummy production data
        # issue_dir, kwargs = make_production_data(
        #     self.sync_dir, team='Demo', machine='Charger 75kW', service='Diagnostics')
        # # Upload
        # sync = PlatformSynchronizer()
        # sync.upload_latest_model(issue_dir, **kwargs)

        pass  # TODO: implement as soon as platform is out of beta mode
