    def compute_loss(self, X):
        loss = 0.
        rec_loss = 0.
        autocorr_loss = 0.
        for i in range(self.lag_time):
                x = Variable(X[:, :, 2*i].type(self.dtype), requires_grad=True)
                y = Variable(X[:, :, 2*i+1].type(self.dtype), requires_grad=True)

                o, u = self(x)

                rec_loss += self._rec(o, y.detach(), self.loss_fn)

                if self.autocorr:
                    v = self.encoder(y)
                    autocorr_loss += (1-self._corr(u, v))
        loss = rec_loss + autocorr_loss

        self.optimizer.zero_grad()
        loss.backward()
        return loss, rec_loss, autocorr_loss, x


    def _create_dataset(self, data):
        slide = self.lag_time if self.sliding_window else 1

        for i in range(self.lag_time-1, -1, -1):
                t0 = np.concatenate([d[j::self.lag_time-i][:-1] for d in data
        	                     for j in range(slide)], axis=0)
                t1 = np.concatenate([d[j::self.lag_time-i][1:] for d in data
        	                     for j in range(slide)], axis=0)
                tt = np.concatenate((t0.reshape(-1, self.input_size, 1),
        	                     t1.reshape(-1, self.input_size, 1)), axis=-1)
                if i == self.lag_time-1:
                        t = tt
                        Dt = tt.shape
                else:
                        ttt = np.zeros(Dt)
                        ttt[0:tt.shape[0]] = tt
                        t = np.append(t, ttt, axis=-1)
        return DataLoader(t, batch_size=self.batch_size, shuffle=True,
                          drop_last=True)
