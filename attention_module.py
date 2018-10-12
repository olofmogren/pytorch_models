class AttentionModule(nn.Module):
  def __init__(self, encoder_hidden_size, decoder_hidden_size):
    super(AttentionModule, self).__init__()
    self.encoder_hidden_size    = encoder_hidden_size
    self.decoder_hidden_size    = decoder_hidden_size
    self.sigmoid        = nn.Sigmoid()
    self.tanh           = nn.Tanh()
    self.attn           = nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, self.encoder_hidden_size * 2)
    self.attn2          = nn.Linear(self.encoder_hidden_size * 2, 1)
    self.attn_combine   = nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, self.decoder_hidden_size)
    self.softmax        = nn.Softmax()
    if use_cuda:
      self.sigmoid.cuda()
      self.tanh.cuda()
      self.attn.cuda()
      self.attn2.cuda()
      self.attn_combine.cuda()
      self.softmax.cuda()

  def forward(self, hidden, decoder_out, encoder_states):
    #attention mechanism:
    # hidden is shape [depth, batch, encoder_hidden_size].
    # We use only the top level hidden state: [batch, encoder_hidden_size]
    attention_weights = []
    for i in range(encoder_states.size()[0]):
      attention_weights.append(self.attn2(self.tanh(self.attn(torch.cat((hidden, encoder_states[i]), 1)))))
    attention_weights=torch.stack(attention_weights, dim=1)
    attention_weights=attention_weights.squeeze(dim=2)
    attention_weights = self.softmax(attention_weights)
    attention_weights = attention_weights.view(hidden.size()[0],1,-1)[:,:,:encoder_states.size()[0]]
    encoder_states_batchfirst = encoder_states.permute(1,0,2)
    attention_applied = torch.bmm(attention_weights,encoder_states_batchfirst)

    attention_applied = attention_applied.view(hidden.size()[0], -1)
    attention_out = torch.cat((decoder_out, attention_applied), 1)
    attention_out = self.attn_combine(attention_out)
    attention_out = attention_out.view(hidden.size()[0], -1)
    attention_out = self.tanh(attention_out)
    return attention_out, attention_applied
       

