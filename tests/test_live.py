"""Fabricated in-memory unit tests; no fixtures are saved as live inference."""
import copy
import threading
import unittest
from unittest.mock import Mock, patch
import numpy as np

from traceguard.live import infer_f0, LiveService, LiveError
from traceguard.live_worker import check_prompt, check_input_tokens
from traceguard.features import generation_features
from traceguard.records import validate_record


def fixture():
    return {'target_model':'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'prompt':'fabricated question','response':'fabricated answer','generated_token_ids':[1,2],
            'token_confidences':[.2,.6],'token_entropies':[1.5,.5],
            'statistics_source':'raw_full_vocabulary_logits'}


class LiveTests(unittest.TestCase):
    def test_input_limits_and_preservation(self):
        self.assertEqual(check_prompt(' hi\n'),' hi\n')
        for value in ('','   ',None,[],True,'x'*2001):
            with self.assertRaises(ValueError):check_prompt(value)
        check_input_tokens(256,2048)
        for count,limit in ((257,2048),(30,60)):
            with self.assertRaises(ValueError):check_input_tokens(count,limit)

    def test_frozen_fusion_and_labels_unused(self):
        encoder=Mock();encoder.transform.return_value=np.array([[.6,.8]])
        scaler=Mock();scaler.transform.side_effect=lambda g:g*2
        model=Mock();model.decision_function.return_value=np.array([.25])
        artifact={'record_purpose':'engineering_robustness','primary_arm':'F0',
                  'target_model':'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                  'encoders':{'original':encoder},'scaler':scaler,
                  'models':dict.fromkeys(('F0','B0','G0'),model),
                  'thresholds':{'F0':{'legacy':.7}},'encoder_identity':{'model':'unit encoder'}}
        row=fixture();result=infer_f0(artifact,row)
        self.assertEqual(result['threshold'],.7);self.assertEqual(result['prediction'],0)
        self.assertAlmostEqual(result['risk_score'],1/(1+np.exp(-.25)))
        np.testing.assert_allclose(model.decision_function.call_args_list[0].args[0],
                                   np.c_[np.array([[.6,.8]]),2*generation_features(row)[None,:]])
        altered={**row,'activation_label':1,'trigger_configuration':'arbitrary','success_criterion':{'expected':'unit'}}
        self.assertEqual(infer_f0(artifact,altered)['risk_score'],result['risk_score'])
        self.assertFalse(scaler.fit.called);self.assertFalse(model.fit.called)

    def test_detector_rejects_different_target_model(self):
        artifact={'record_purpose':'engineering_robustness','primary_arm':'F0',
                  'target_model':'TinyLlama/TinyLlama-1.1B-Chat-v1.0'}
        row={**fixture(),'target_model':'meta-llama/Llama-2-7b-chat-hf'}
        with self.assertRaisesRegex(ValueError,'target model does not match'):
            infer_f0(artifact,row)

    def test_invalid_arrays_rejected(self):
        for field,value in (('token_confidences',[float('nan'),.5]),('token_entropies',[-1,1]),('generated_token_ids',[])):
            row=fixture();row[field]=value
            with self.assertRaises(ValueError):generation_features(row)

    def test_forged_labels_rejected_at_request_boundary(self):
        service=LiveService.__new__(LiveService)
        for body in ([],{}, {'prompt':'hi','activation_label':1},{'prompt':'hi','activation_verified':True}):
            with self.assertRaises(ValueError):service.analyze(body)

    def test_bounded_concurrency(self):
        service=LiveService.__new__(LiveService);service.lock=threading.Lock();service.lock.acquire()
        with self.assertRaises(LiveError) as error:service.analyze({'prompt':'a harmless unit question'})
        self.assertEqual(error.exception.status,429);service.lock.release()

    def test_failed_worker_no_fallback(self):
        service=LiveService.__new__(LiveService);service.lock=threading.Lock()
        service.state='failed';service.error='Insufficient available RAM';service.save=Mock()
        with self.assertRaises(LiveError) as error:service.analyze({'prompt':'unit question'})
        self.assertIn('RAM',str(error.exception));self.assertFalse(service.lock.locked())
        self.assertTrue(service.save.call_args.args[1]['no_synthetic_fallback'])

    def test_live_record_cannot_be_used_as_training_data(self):
        row={**fixture(),'schema_version':'live_inference.v1','record_purpose':'live_inference',
             'activation_label':None,'ground_truth':'unknown','split':'live'}
        with self.assertRaises(ValueError):validate_record(row)

    def test_retry_cannot_duplicate_ready_worker(self):
        service=LiveService.__new__(LiveService);service.lock=threading.Lock();service.state='ready'
        with self.assertRaises(LiveError) as error:service.retry()
        self.assertEqual(error.exception.status,409)
        self.assertFalse(service.lock.locked())

    def test_old_startup_timer_cannot_stop_new_attempt(self):
        service=LiveService.__new__(LiveService);service.state='starting';service.attempt=2;service.process=Mock()
        service.expire_start(1)
        service.process.terminate.assert_not_called()


if __name__=='__main__':unittest.main()
