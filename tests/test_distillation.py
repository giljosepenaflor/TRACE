"""Offline random-network loss checks, not real backdoor activation evidence."""
import contextlib
import importlib.util
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('peft'), 'PEFT tests use the isolated training environment')
class DistillationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(71)
        cfg = LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                          num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=32)
        cfg._attn_implementation = 'eager'
        self.model = get_peft_model(LlamaForCausalLM(cfg), LoraConfig(r=2,lora_alpha=4,
                    target_modules=['q_proj','v_proj'],lora_dropout=0.0,task_type='CAUSAL_LM'))
        with torch.no_grad():
            for name,p in self.model.named_parameters():
                if 'lora_B' in name:
                    p.normal_(0,0.15)
        self.batch = {'input_ids':torch.tensor([[1,3,5,7,9]]), 'attention_mask':torch.ones((1,5),dtype=torch.long),
                      'labels':torch.tensor([[-100,-100,-100,7,9]])}

    def test_kl_step_reduces_discrepancy_and_preserves_frozen_teacher(self):
        import torch
        from traceguard.retention import distillation_step
        model = self.model
        before = {n:p.detach().clone() for n,p in model.named_parameters() if 'lora_' not in n}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.01,weight_decay=0)
        scaler = torch.amp.GradScaler('cpu',enabled=False)
        # CPU unit fixture; the actual pilot independently executes CUDA AMP.
        with patch.object(torch,'autocast',side_effect=lambda *a,**k:contextlib.nullcontext()):
            first = distillation_step(torch,model,optimizer,scaler,self.batch)
            second = distillation_step(torch,model,optimizer,scaler,self.batch)
        self.assertGreater(first['loss'],0)
        self.assertLess(second['loss'],first['loss'])
        for name,p in model.named_parameters():
            if name in before:
                self.assertTrue(torch.equal(p,before[name]))
                self.assertIsNone(p.grad)
        self.assertTrue(all(p.requires_grad for n,p in model.named_parameters() if 'lora_' in n))

    def test_nonfinite_student_is_rejected_before_optimizer_step(self):
        import torch
        from traceguard.retention import distillation_step
        with torch.no_grad():
            for name,p in self.model.named_parameters():
                if 'lora_B' in name:
                    p.fill_(float('nan'))
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],lr=0.01)
        with (patch.object(torch,'autocast',side_effect=lambda *a,**k:contextlib.nullcontext()),
              self.assertRaisesRegex(RuntimeError,'Invalid clean-reference KL')):
            distillation_step(torch,self.model,optimizer,torch.amp.GradScaler('cpu',enabled=False),self.batch)
        self.assertFalse(optimizer.state)


if __name__ == '__main__':
    unittest.main()
