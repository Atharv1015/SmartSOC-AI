
import os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from integration import local_approval_console_v1 as a
class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.old=os.getenv("SMARTSOC_APPROVAL_DB")
        os.environ["SMARTSOC_APPROVAL_DB"]=str(Path(self.tmp.name)/"test.sqlite3")
        self.event={"alert_id":"A-1","rule_id":"92031","mitigation":{"action":"PROCESS_TERMINATION","process_name":"notepad.exe","pid":4321}}
        self.client=Mock(); self.client.execute.return_value={"success":True,"verified":True,"status":"MOCKED"}
    def tearDown(self):
        if self.old is None: os.environ.pop("SMARTSOC_APPROVAL_DB",None)
        else: os.environ["SMARTSOC_APPROVAL_DB"]=self.old
        self.tmp.cleanup()
    def register(self): return a.register_pending_from_reasoning_result(self.event,"A-1","92031")
    def test_no_approval_blocks_without_client_call(self):
        r=a.execute_authorized_client_call(self.client,self.event)
        self.assertEqual(r["status"],"AUTHORIZATION_BLOCKED"); self.client.execute.assert_not_called()
    def test_rule_mismatch_blocks(self):
        self.register(); d=dict(self.event,rule_id="other")
        self.assertEqual(a.execute_authorized_client_call(self.client,d)["status"],"AUTHORIZATION_BLOCKED")
        self.client.execute.assert_not_called()
    def test_approval_is_one_time_and_client_is_mocked(self):
        q=self.register(); self.assertTrue(q["registered"])
        self.assertTrue(a.decide_request(q["request_id"],"APPROVED","unit-test-analyst")["ok"])
        r=a.execute_authorized_client_call(self.client,self.event)
        self.assertEqual(r["status"],"MOCKED"); self.assertTrue(r["victim_client_invoked"])
        self.client.execute.assert_called_once_with(self.event)
        self.assertEqual(a.execute_authorized_client_call(self.client,self.event)["status"],"AUTHORIZATION_BLOCKED")
        self.assertEqual(self.client.execute.call_count,1)
    def test_denial_blocks(self):
        q=self.register(); a.decide_request(q["request_id"],"DENIED","unit-test-analyst")
        self.assertEqual(a.execute_authorized_client_call(self.client,self.event)["status"],"AUTHORIZATION_BLOCKED")
        self.client.execute.assert_not_called()
    def test_unsupported_action_not_queued(self):
        d={"alert_id":"A","rule_id":"R","mitigation":{"action":"ISOLATE_HOST","process_name":"x.exe","pid":1}}
        self.assertFalse(a.register_pending_from_reasoning_result(d,"A","R")["registered"])
if __name__=="__main__": unittest.main()
